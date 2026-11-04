#!/usr/bin/env bash
# The project's quality gates. CI runs exactly this script, so a local run reproduces CI.
# Usage (inside the project's virtual environment): bash scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== dependency consistency";  python -m pip check
echo "== lint";                    python -m ruff check .
echo "== format";                  python -m ruff format --check .

echo "== catalog validation (valid catalogs)"
python -m recommender.cli validate-data --input data/sample/products.jsonl
python -m recommender.cli validate-data --input data/demo/products.jsonl

echo "== catalog validation must FAIL on an intentionally invalid catalog"
if python -m recommender.cli validate-data --input tests/fixtures/invalid_catalog.jsonl >/dev/null 2>&1; then
  echo "ERROR: the invalid catalog was accepted" >&2
  exit 1
fi
echo "rejected as expected"

echo "== tests (no network, no model download, no live service)"
python -m pytest -m "not slow" -q

echo "== evaluation tooling smoke test (development split)"
python -m recommender.cli evaluate --split dev --runs 1 --output "$(mktemp -d)/eval-smoke.json" >/dev/null
echo "all checks passed"
