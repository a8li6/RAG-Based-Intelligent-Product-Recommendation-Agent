"""Measure end-to-end latency for matched Persian/English request pairs (same intent and ids).

A like-for-like comparison: the two evaluation reports cover different query texts and separate
runs, so their p50 values must not be compared directly.

Usage: python scripts/measure_latency_pairs.py    (writes reports/fa/latency-pairs.json)
"""

from __future__ import annotations

import json
import platform
import statistics as st
import tempfile
import time
from pathlib import Path

from recommender.catalog import load_catalog
from recommender.config import IsolatedSettings
from recommender.index import build_artifacts
from recommender.local_embedder import LocalEmbedder
from recommender.schemas import Filters, RecommendationRequest
from recommender.service import create_dependencies, recommend

ROOT = Path(__file__).resolve().parents[1]
RUNS, WARMUP = 60, 5


def load(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {r["query_id"]: r for r in map(json.loads, (x for x in lines if x.strip()))}


def median_ms(deps, rec: dict) -> float:
    controls = rec["controls"]
    request = RecommendationRequest(
        query=rec["query"], filters=Filters(**controls) if controls else None
    )
    for _ in range(WARMUP):
        recommend(request, None, deps)
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        recommend(request, None, deps)
        times.append((time.perf_counter() - start) * 1000)
    return st.median(times)


def main() -> None:
    catalog = load_catalog(ROOT / "data" / "demo" / "products.jsonl")
    persian = load(ROOT / "data" / "eval" / "fa" / "queries.jsonl")
    english = load(ROOT / "data" / "eval" / "queries.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        build_artifacts(catalog, LocalEmbedder(), Path(tmp) / "index")
        settings = IsolatedSettings(
            artifact_dir=Path(tmp) / "index", catalog_path=ROOT / "data" / "demo" / "products.jsonl"
        )
        deps = create_dependencies(settings, LocalEmbedder())
        pairs = []
        for qid in sorted(persian, key=lambda x: (x[0], int(x[1:]))):
            pairs.append(
                {
                    "query_id": qid,
                    "expected_status": persian[qid]["expected_status"],
                    "persian_ms": median_ms(deps, persian[qid]),
                    "english_ms": median_ms(deps, english[qid]),
                }
            )
    answerable = [p for p in pairs if p["expected_status"] == "ok"]
    ratios = [p["persian_ms"] / p["english_ms"] for p in answerable]
    quartiles = st.quantiles(ratios, n=4)
    report = {
        "method": f"per-query median of {RUNS} warm calls ({WARMUP} warm-up), in-process, answerable queries only",
        "hardware": f"{platform.platform()} / Python {platform.python_version()}",
        "pairs": len(answerable),
        "persian_median_ms": st.median(p["persian_ms"] for p in answerable),
        "english_median_ms": st.median(p["english_ms"] for p in answerable),
        "ratio_median": st.median(ratios),
        "ratio_q1": quartiles[0],
        "ratio_q3": quartiles[2],
        "per_query": pairs,
    }
    out = ROOT / "reports" / "fa" / "latency-pairs.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{len(answerable)} pairs: Persian {report['persian_median_ms']:.2f} ms, English {report['english_median_ms']:.2f} ms, "
        f"ratio median {report['ratio_median']:.2f} (IQR {report['ratio_q1']:.2f}-{report['ratio_q3']:.2f})"
    )


if __name__ == "__main__":
    main()
