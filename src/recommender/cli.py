"""Command-line entry points. Subcommands are added milestone by milestone."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from recommender.catalog import CatalogValidationError, catalog_report, load_catalog
from recommender.config import ConfigError, Settings, load_settings
from recommender.embeddings import E5Embedder, Embedder
from recommender.index import ArtifactError, build_artifacts, is_pinned_revision, load_artifacts
from recommender.llm import Explainer
from recommender.query import InvalidQueryError
from recommender.retrieval import Retriever
from recommender.schemas import Constraints, Filters, RecommendationRequest
from recommender.service import create_dependencies, recommend

EmbedderFactory = Callable[[Settings], Embedder]


def default_embedder_factory(settings: Settings) -> Embedder:
    return E5Embedder(
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_device,
        settings.embedding_batch_size,
    )


def _safe_output(path: str, source: str, label: str) -> Path | None:
    """Outputs must never overwrite the original source file."""
    target = Path(path)
    if target.resolve() == Path(source).resolve():
        print(f"error: --{label} must not be the input file ({source})", file=sys.stderr)
        return None
    return target


def cmd_validate_data(args: argparse.Namespace) -> int:
    report_path = canonical_path = None
    if args.report and (report_path := _safe_output(args.report, args.input, "report")) is None:
        return 2
    if (
        args.canonical_out
        and (canonical_path := _safe_output(args.canonical_out, args.input, "canonical-out"))
        is None
    ):
        return 2
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
    if report_path is not None:
        report_path.write_text(
            json.dumps(catalog_report(catalog, args.input), indent=2) + "\n", encoding="utf-8"
        )
    if canonical_path is not None:
        canonical_path.write_bytes(catalog.canonical_bytes())
    print(f"OK: {len(catalog)} records, sha256={catalog.sha256}")
    return 0


def cmd_build_index(
    args: argparse.Namespace,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    explainer: Explainer | None = None,
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
    manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    if not is_pinned_revision(manifest["embedding_revision"]):
        print(
            f"warning: embedding revision '{manifest['embedding_revision']}' is not an immutable "
            "commit hash, so this build is not reproducible; pin EMBEDDING_REVISION before any "
            "benchmark or release",
            file=sys.stderr,
        )
    print(
        f"OK: built {len(catalog)} products into {built} (catalog sha256={catalog.sha256}, "
        f"embedding {manifest['embed_seconds']}s, total {manifest['build_seconds']}s)"
    )
    return 0


def cmd_search(
    args: argparse.Namespace,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    explainer: Explainer | None = None,
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


def cmd_recommend(
    args: argparse.Namespace,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    explainer: Explainer | None = None,
) -> int:
    """Print the public response contract (the same JSON the API returns)."""
    update: dict = {}
    if args.artifacts:
        update["artifact_dir"] = Path(args.artifacts)
    if args.no_llm:
        update["explanation_mode"] = "deterministic"
    settings = settings.model_copy(update=update)
    try:
        request = RecommendationRequest(
            query=args.query,
            filters=Filters(
                max_price_cents=args.max_price_cents,
                min_ram_gb=args.min_ram_gb,
                max_weight_kg=args.max_weight_kg,
                require_in_stock=False if args.include_out_of_stock else None,
            ),
            requested_count=args.count,
        )
        deps = create_dependencies(settings, embedder_factory(settings), explainer)
        result = recommend(request, None, deps)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (InvalidQueryError, ValueError) as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return 2
    print(result.response.model_dump_json(indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="validate a JSONL product catalog")
    validate.add_argument("--input", required=True, help="path to products.jsonl")
    validate.add_argument("--report", help="write a JSON validation report to this path")
    validate.add_argument(
        "--canonical-out", help="write the deterministic ID-sorted canonical snapshot here"
    )
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

    rec = sub.add_parser("recommend", help="run the full workflow and print the response JSON")
    rec.add_argument("--query", required=True)
    rec.add_argument("--artifacts", help="artifact directory (default: ARTIFACT_DIR)")
    rec.add_argument("--max-price-cents", type=int)
    rec.add_argument("--min-ram-gb", type=int)
    rec.add_argument("--max-weight-kg", type=float)
    rec.add_argument("--include-out-of-stock", action="store_true")
    rec.add_argument("--count", type=int, default=3, help="number of products, 1-3")
    rec.add_argument("--no-llm", action="store_true", help="force deterministic explanations")
    rec.set_defaults(func=cmd_recommend)
    return parser


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: EmbedderFactory | None = None,
    explainer: Explainer | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.func is cmd_validate_data:
        return cmd_validate_data(args)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return args.func(args, settings, embedder_factory or default_embedder_factory, explainer)


if __name__ == "__main__":
    raise SystemExit(main())
