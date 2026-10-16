"""Command-line entry points. Subcommands are added milestone by milestone."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from recommender.catalog import CatalogValidationError, load_catalog
from recommender.config import ConfigError, Settings, load_settings
from recommender.embeddings import E5Embedder, Embedder
from recommender.index import ArtifactError, build_artifacts, load_artifacts
from recommender.retrieval import Retriever
from recommender.schemas import Constraints

EmbedderFactory = Callable[[Settings], Embedder]


def default_embedder_factory(settings: Settings) -> Embedder:
    return E5Embedder(
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_device,
        settings.embedding_batch_size,
    )


def cmd_validate_data(args: argparse.Namespace) -> int:
    try:
        catalog = load_catalog(args.input)
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as exc:
        print(f"error: {args.input} is not valid UTF-8: {exc.reason}", file=sys.stderr)
        return 2
    except CatalogValidationError as exc:
        print(f"validation failed: {len(exc.issues)} error(s) in {args.input}", file=sys.stderr)
        for issue in exc.issues:
            print(f"  {issue}", file=sys.stderr)
        return 1
    print(f"OK: {len(catalog)} records, sha256={catalog.sha256}")
    return 0


def cmd_build_index(
    args: argparse.Namespace, settings: Settings, embedder_factory: EmbedderFactory
) -> int:
    try:
        catalog = load_catalog(args.input)
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2
    except CatalogValidationError as exc:
        print(f"validation failed: {len(exc.issues)} error(s)", file=sys.stderr)
        for issue in exc.issues:
            print(f"  {issue}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else settings.artifact_dir
    try:
        built = build_artifacts(catalog, embedder_factory(settings), output)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:  # model/IO/contract failures
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(f"OK: built {len(catalog)} products into {built} (catalog sha256={catalog.sha256})")
    return 0


def cmd_search(
    args: argparse.Namespace, settings: Settings, embedder_factory: EmbedderFactory
) -> int:
    directory = Path(args.artifacts) if args.artifacts else settings.artifact_dir
    try:
        artifacts = load_artifacts(directory)
        retriever = Retriever(artifacts, embedder_factory(settings), settings.retrieval_k)
        constraints = Constraints(
            max_price_cents=args.max_price_cents,
            min_ram_gb=args.min_ram_gb,
            max_weight_kg=args.max_weight_kg,
            require_in_stock=not args.include_out_of_stock,
        )
        rows = retriever.eligible_rows(constraints)
        candidates = retriever.search(args.query, rows)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"eligible: {len(rows)} of {len(artifacts.catalog)} products")
    for candidate in candidates:
        title = artifacts.catalog.products[candidate.row].title
        print(f"{candidate.product_id}  {candidate.score:.4f}  {title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="validate a JSONL product catalog")
    validate.add_argument("--input", required=True, help="path to products.jsonl")
    validate.set_defaults(func=cmd_validate_data)

    build = sub.add_parser("build-index", help="embed the catalog and write index artifacts")
    build.add_argument("--input", required=True, help="path to products.jsonl")
    build.add_argument("--output", help="artifact directory (default: ARTIFACT_DIR)")
    build.set_defaults(func=cmd_build_index)

    search = sub.add_parser("search", help="debug: show eligible candidates and scores")
    search.add_argument("--query", required=True)
    search.add_argument("--artifacts", help="artifact directory (default: ARTIFACT_DIR)")
    search.add_argument("--max-price-cents", type=int)
    search.add_argument("--min-ram-gb", type=int)
    search.add_argument("--max-weight-kg", type=float)
    search.add_argument("--include-out-of-stock", action="store_true")
    search.set_defaults(func=cmd_search)
    return parser


def main(argv: Sequence[str] | None = None, embedder_factory: EmbedderFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.func is cmd_validate_data:
        return cmd_validate_data(args)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return args.func(args, settings, embedder_factory or default_embedder_factory)


if __name__ == "__main__":
    raise SystemExit(main())
