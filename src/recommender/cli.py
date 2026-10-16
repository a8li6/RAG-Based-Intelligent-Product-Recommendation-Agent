"""Command-line entry points. Subcommands are added milestone by milestone."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from recommender.catalog import CatalogValidationError, load_catalog


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="validate a JSONL product catalog")
    validate.add_argument("--input", required=True, help="path to products.jsonl")
    validate.set_defaults(func=cmd_validate_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
