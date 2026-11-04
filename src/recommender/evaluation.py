"""Reproducible evaluation: frozen splits, explicit denominators, baselines, raw per-query output.

Retrieval quality and end-to-end behavior are measured separately, and each retrieval method is
compared on the same eligible products:

- builtin     the application's retriever (built-in embedder) over the eligible subset
- keyword     BM25 over the same documents and the same eligible subset
- random      a seeded random order of the eligible subset (the floor)
- post_filter global top-k over the whole catalog, filtered afterwards (the bug the design
              avoids; shown as an ablation)

Generation (templates or an optional LLM) receives no credit for retrieval it did not improve:
retrieval metrics never look at explanations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import resource
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import faiss

from recommender.catalog import Catalog, is_eligible, load_catalog
from recommender.config import IsolatedSettings
from recommender.documents import DOCUMENT_FORMAT_VERSION, catalog_documents
from recommender.index import build_artifacts
from recommender.local_embedder import LocalEmbedder, tokenize
from recommender.metrics import (
    mean_and_count,
    mrr_at_k,
    ndcg_at_k,
    oracle_recall_at_k,
    percentile,
    rate,
    recall_at_k,
    relevant_eligible,
)
from recommender.query import parse_and_merge
from recommender.schemas import (
    Constraints,
    Filters,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from recommender.service import Dependencies, create_dependencies, recommend

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = ROOT / "data" / "eval" / "queries.jsonl"
DEFAULT_META = ROOT / "data" / "eval" / "meta.json"
DEFAULT_CATALOG = ROOT / "data" / "demo" / "products.jsonl"
RANKERS = ("builtin", "keyword", "random", "post_filter")
RANDOM_DRAWS = 50
STAGES = ("parse", "filter", "retrieve", "select", "explain", "render")


class EvaluationError(Exception):
    """The evaluation cannot be trusted (label drift, frozen labels changed, bad input)."""


# --- loading and integrity ------------------------------------------------------------------


def frozen_test_labels_digest(records: list[dict]) -> str:
    frozen = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["query_id"])
    payload = json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_eval_set(
    queries_path: Path = DEFAULT_QUERIES, meta_path: Path = DEFAULT_META, *, catalog: Catalog
) -> tuple[list[dict], dict, str]:
    """Load the set and verify that labels still describe this catalog and test labels are frozen."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    raw = Path(queries_path).read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if meta["catalog_sha256"] != catalog.sha256:
        raise EvaluationError(
            "label drift: the evaluation set was built for catalog "
            f"{meta['catalog_sha256'][:12]} but the catalog is {catalog.sha256[:12]}"
        )
    if frozen_test_labels_digest(records) != meta["test_labels_sha256"]:
        raise EvaluationError("the frozen test labels have changed since they were recorded")
    return records, meta, hashlib.sha256(raw).hexdigest()


# --- keyword baseline ------------------------------------------------------------------------


class Bm25:
    """BM25 over the same product documents, without stemming or the concept lexicon."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in documents]
        self.avg_len = sum(len(t) for t in self.tokens) / len(self.tokens)
        df: dict[str, int] = defaultdict(int)
        for toks in self.tokens:
            for term in set(toks):
                df[term] += 1
        n = len(documents)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: str, row: int) -> float:
        toks = self.tokens[row]
        total = 0.0
        for term in tokenize(query):
            tf = toks.count(term)
            if tf and term in self.idf:
                norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
                total += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return total

    def rank(self, query: str, rows: list[int], k: int) -> list[int]:
        return sorted(rows, key=lambda r: (-self.score(query, r), r))[:k]


# --- evidence audit --------------------------------------------------------------------------

_MONEY = r"(\d+\.\d{2}) USD"
_REASON_PATTERNS = [
    (re.compile(rf"^Priced at {_MONEY}, within your maximum of {_MONEY}\.$"), "price"),
    (re.compile(r"^Has (\d+) GB RAM, meeting your minimum of (\d+) GB\.$"), "ram"),
    (re.compile(r"^Weighs (\d+\.\d{2}) kg, within your limit of ([\d.]+) kg\.$"), "weight"),
    (re.compile(r"^Marked in stock in this catalog snapshot\.$"), "stock"),
    (re.compile(r"^The catalog lists use cases related to your request: (.+)\.$"), "use_cases"),
    (
        re.compile(
            r"^The catalog lists (RAM|storage|weight|GPU|price|stock status|use cases): (.+)\.$"
        ),
        "fact",
    ),
]


def _money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def audit_item(item: RecommendationItem, catalog: Catalog) -> tuple[int, int]:
    """Independently re-check every displayed fact and rendered reason against the catalog.

    Returns (claims_checked, claims_valid). Unknown reason sentences count as invalid.
    """
    product = catalog.get(item.product_id)
    checks: list[bool] = [
        item.title == product.title,
        item.price_cents == product.price_cents,
        item.currency == product.currency,
        item.in_stock == product.in_stock,
        item.source_url == product.source_url,
        item.attributes.get("ram_gb") == product.ram_gb,
        item.attributes.get("storage_gb") == product.storage_gb,
        item.attributes.get("weight_kg") == product.weight_kg,
        item.attributes.get("gpu_type") == product.gpu_type,
        item.attributes.get("use_cases") == product.use_cases,
    ]
    for ref in item.evidence_refs:
        pid, _, field = ref.partition(".")
        checks.append(pid == item.product_id and field in {
            "price_cents", "ram_gb", "storage_gb", "weight_kg", "gpu_type", "in_stock", "use_cases"
        })  # fmt: skip
    for text in item.reasons:
        checks.append(_reason_is_valid(text, product))
    return len(checks), sum(checks)


def _reason_is_valid(text: str, product) -> bool:
    for pattern, kind in _REASON_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        if kind == "price":
            return m.group(1) == _money(product.price_cents)
        if kind == "ram":
            return product.ram_gb is not None and int(m.group(1)) == product.ram_gb
        if kind == "weight":
            return product.weight_kg is not None and m.group(1) == f"{product.weight_kg:.2f}"
        if kind == "stock":
            return product.in_stock
        if kind == "use_cases":
            return set(m.group(1).split(", ")) <= set(product.use_cases)
        label, value = m.group(1), m.group(2)
        expected = {
            "RAM": f"{product.ram_gb} GB",
            "storage": f"{product.storage_gb} GB",
            "weight": f"{product.weight_kg:.2f} kg" if product.weight_kg is not None else None,
            "GPU": product.gpu_type,
            "price": f"{_money(product.price_cents)} USD",
            "stock status": str(product.in_stock),
            "use cases": ", ".join(product.use_cases),
        }[label]
        return expected is not None and value == expected
    return False


# --- per-query evaluation -------------------------------------------------------------------


def _interpretation_ok(response: RecommendationResponse, constraints: dict) -> bool:
    got = response.interpreted_filters
    return (
        got.max_price_cents == constraints["max_price_cents"]
        and got.min_ram_gb == constraints["min_ram_gb"]
        and got.max_weight_kg == constraints["max_weight_kg"]
        and got.require_in_stock == constraints["require_in_stock"]
    )


def _semantic_query(rec: dict) -> str:
    """The text the retriever would embed (constraint phrases removed by the parser)."""
    intent = parse_and_merge(
        rec["query"], Filters(**rec["controls"]) if rec["controls"] else None
    ).intent
    return intent.semantic_query


def _rank_all(
    rec: dict, deps: Dependencies, bm25: Bm25
) -> tuple[dict[str, list[str]], float | None]:
    """Top-10 product IDs per retrieval method, over the ground-truth eligible subset."""
    catalog = deps.catalog
    ids = catalog.product_ids
    rows = [ids.index(p) for p in rec["eligible_ids"]]
    semantic = _semantic_query(rec)
    k = deps.settings.retrieval_k

    built = deps.retriever.search(semantic, rows)
    top_score = built[0].score if built else None
    rankings = {"builtin": [c.product_id for c in built]}
    rankings["keyword"] = [ids[r] for r in bm25.rank(semantic, rows, k)]

    rng = random.Random(f"eval:{rec['query_id']}")
    rankings["random"] = [ids[r] for r in rng.sample(rows, min(k, len(rows)))] if rows else []

    everything = list(range(len(ids)))
    eligible = set(rec["eligible_ids"])
    global_top = [c.product_id for c in deps.retriever.search(semantic, everything)]
    rankings["post_filter"] = [p for p in global_top if p in eligible]
    return rankings, top_score


def _random_expectation(rec: dict, ids: list[str], k: int, relevant: set[str]) -> dict:
    """Average the random-order metrics over many draws (one draw is too noisy)."""
    rows = [ids.index(p) for p in rec["eligible_ids"]]
    totals = {"recall_at_10": [], "mrr_at_3": [], "ndcg_at_3": []}
    for draw in range(RANDOM_DRAWS):
        rng = random.Random(f"eval:{rec['query_id']}:{draw}")
        ranked = [ids[r] for r in rng.sample(rows, min(k, len(rows)))] if rows else []
        totals["recall_at_10"].append(recall_at_k(ranked, relevant, 10))
        totals["mrr_at_3"].append(mrr_at_k(ranked, relevant, 3))
        totals["ndcg_at_3"].append(ndcg_at_k(ranked, rec["relevance"], rec["eligible_ids"], 3))
    return {
        name: (None if vals[0] is None else sum(vals) / len(vals)) for name, vals in totals.items()
    }


def _retrieval_metrics(rec: dict, ranked: list[str], relevant: set[str]) -> dict:
    return {
        "recall_at_10": recall_at_k(ranked, relevant, 10),
        "mrr_at_3": mrr_at_k(ranked, relevant, 3),
        "ndcg_at_3": ndcg_at_k(ranked, rec["relevance"], rec["eligible_ids"], 3),
    }


def _timed(deps: Dependencies, request: RecommendationRequest):
    start = time.perf_counter()
    result = recommend(request, None, deps)
    return result, (time.perf_counter() - start) * 1000


def evaluate_records(
    records: list[dict], deps: Dependencies, *, runs: int = 5
) -> tuple[list[dict], dict]:
    """Run every query through the full system and the retrieval baselines."""
    catalog = deps.catalog
    bm25 = Bm25(catalog_documents(catalog))
    ids = catalog.product_ids
    per_query: list[dict] = []
    latencies: list[float] = []
    stage_ms: dict[str, list[float]] = defaultdict(list)
    cold_ms: float | None = None

    for rec in records:
        constraints = Constraints(**rec["constraints"])
        request = RecommendationRequest(
            query=rec["query"],
            filters=Filters(**rec["controls"]) if rec["controls"] else None,
        )
        result, ms = _timed(deps, request)  # the very first request is the cold one
        if cold_ms is None:
            cold_ms = ms
        for _ in range(runs):  # warm timings
            warm, ms = _timed(deps, request)
            latencies.append(ms)
            for stage in STAGES:
                if stage in warm.response.timings_ms:
                    stage_ms[stage].append(warm.response.timings_ms[stage])

        response = result.response
        returned = [i.product_id for i in response.recommendations]
        violations = [pid for pid in returned if not is_eligible(catalog.get(pid), constraints)]
        claims = [audit_item(i, catalog) for i in response.recommendations]
        relevant = relevant_eligible(rec["relevance"], rec["eligible_ids"])
        entry = {
            "query_id": rec["query_id"],
            "split": rec["split"],
            "category": rec["category"],
            "query": rec["query"],
            "expected_status": rec["expected_status"],
            "status": response.status,
            "returned": returned,
            "violations": violations,
            "interpretation_ok": _interpretation_ok(response, rec["constraints"]),
            "interpreted": response.interpreted_filters.model_dump(),
            "claims_checked": sum(c for c, _ in claims),
            "claims_valid": sum(v for _, v in claims),
            "explanation_mode": response.explanation_mode,
            "fallback_reason": response.fallback_reason,
            "relevant_eligible": len(relevant),
            "system_mrr_at_3": mrr_at_k(returned, relevant, 3),
            "system_ndcg_at_3": ndcg_at_k(returned, rec["relevance"], rec["eligible_ids"], 3),
            "lexicon_hit_expected": rec["lexicon_hit_expected"],
        }
        if rec["expected_status"] == "ok" or rec["category"] == "irrelevant":
            rankings, top_score = _rank_all(rec, deps, bm25)
            entry["top_score"] = top_score
            entry["rankings"] = rankings
            if rec["expected_status"] == "ok":
                entry["retrieval"] = {
                    name: _retrieval_metrics(rec, ranked, relevant)
                    for name, ranked in rankings.items()
                    if name != "random"
                }
                entry["retrieval"]["random"] = _random_expectation(
                    rec, ids, deps.settings.retrieval_k, relevant
                )
                entry["oracle_recall_at_10"] = oracle_recall_at_k(len(relevant), 10)
        per_query.append(entry)

    operational = {
        "cold_first_request_ms": cold_ms,
        "warm_request_count": len(latencies),
        "warm_p50_ms": percentile(latencies, 50),
        "warm_p95_ms": percentile(latencies, 95),
        "stage_mean_ms": {s: sum(v) / len(v) for s, v in stage_ms.items() if v},
    }
    if operational["stage_mean_ms"]:
        operational["slowest_stage"] = max(
            operational["stage_mean_ms"], key=operational["stage_mean_ms"].get
        )
    return per_query, operational


# --- aggregation -----------------------------------------------------------------------------


def aggregate(per_query: list[dict]) -> dict:
    ok_queries = [q for q in per_query if q["expected_status"] == "ok"]
    positive = [q for q in ok_queries if q["relevant_eligible"] > 0]

    def retrieval(name: str, rows: list[dict]) -> dict:
        return {
            metric: mean_and_count(q["retrieval"][name][metric] for q in rows)
            for metric in ("recall_at_10", "mrr_at_3", "ndcg_at_3")
        }

    retrieval_summary = {name: retrieval(name, positive) for name in RANKERS}
    retrieval_summary["oracle_recall_at_10"] = mean_and_count(
        q["oracle_recall_at_10"] for q in positive
    )

    by_category: dict[str, dict] = {}
    for cat in sorted({q["category"] for q in positive}):
        rows = [q for q in positive if q["category"] == cat]
        by_category[cat] = {name: retrieval(name, rows) for name in ("builtin", "keyword")}
        by_category[cat]["queries"] = len(rows)

    lexicon = {}
    for label, flag in (("lexicon_hit_expected", True), ("lexicon_miss_expected", False)):
        rows = [q for q in positive if q["lexicon_hit_expected"] is flag]
        lexicon[label] = {name: retrieval(name, rows) for name in ("builtin", "keyword")}
        lexicon[label]["queries"] = len(rows)

    returned_items = sum(len(q["returned"]) for q in per_query)
    violating = sum(len(q["violations"]) for q in per_query)
    by_status = defaultdict(lambda: [0, 0])
    for q in per_query:
        by_status[q["category"]][1] += 1
        by_status[q["category"]][0] += q["status"] == q["expected_status"]

    def count(pred, rows) -> tuple[int, int]:
        rows = list(rows)
        return sum(1 for q in rows if pred(q)), len(rows)

    status_rows = [q for q in per_query if q["expected_status"] in ("ok", "no_match")]
    system = {
        "status_accuracy": rate(*count(lambda q: q["status"] == q["expected_status"], per_query)),
        "status_accuracy_by_category": {c: rate(*v) for c, v in sorted(by_status.items())},
        "interpretation_accuracy": rate(*count(lambda q: q["interpretation_ok"], status_rows)),
        "constraint_violation_rate": rate(violating, returned_items),
        "answer_rate_on_positive_queries": rate(
            *count(lambda q: q["status"] == "ok" and q["returned"], positive)
        ),
        "false_empty_rate": rate(*count(lambda q: q["status"] == "no_match", ok_queries)),
        "no_match_accuracy": rate(
            *count(
                lambda q: q["status"] == "no_match",
                (q for q in per_query if q["expected_status"] == "no_match"),
            )
        ),
        "clarification_accuracy": rate(
            *count(
                lambda q: q["status"] == "needs_clarification",
                (q for q in per_query if q["expected_status"] == "needs_clarification"),
            )
        ),
        "abstention_on_irrelevant": rate(
            *count(
                lambda q: q["status"] == "needs_clarification",
                (q for q in per_query if q["category"] == "irrelevant"),
            )
        ),
        "mrr_at_3": mean_and_count(q["system_mrr_at_3"] for q in positive),
        "ndcg_at_3": mean_and_count(q["system_ndcg_at_3"] for q in positive),
        "evidence_validity": rate(
            sum(q["claims_valid"] for q in per_query), sum(q["claims_checked"] for q in per_query)
        ),
        "fallback_rate": rate(
            sum(
                1
                for q in per_query
                if q["status"] == "ok" and q["fallback_reason"] not in (None, "llm_disabled")
            ),
            sum(1 for q in per_query if q["status"] == "ok"),
        ),
    }
    return {
        "queries": len(per_query),
        "positive_queries_with_relevant_eligible": len(positive),
        "retrieval": retrieval_summary,
        "retrieval_by_category": by_category,
        "retrieval_by_lexicon": lexicon,
        "system": system,
    }


# --- calibration -----------------------------------------------------------------------------


def calibrate_min_similarity(scored: list[tuple[float, bool]]) -> dict:
    """Pick the abstention cutoff that maximizes balanced accuracy on DEVELOPMENT data only.

    `scored` holds (best eligible similarity, should_answer). Predict "answer" when
    score >= threshold. Ties go to the lowest threshold (prefer answering).
    """
    if not scored or not any(a for _, a in scored) or all(a for _, a in scored):
        raise EvaluationError("calibration needs both answerable and abstain-expected queries")
    values = sorted({s for s, _ in scored})
    candidates = [
        values[0] - 1e-6,
        *[(a + b) / 2 for a, b in itertools.pairwise(values)],
        values[-1] + 1e-6,
    ]
    positives = [s for s, a in scored if a]
    negatives = [s for s, a in scored if not a]
    best = None
    for t in candidates:
        tpr = sum(s >= t for s in positives) / len(positives)
        tnr = sum(s < t for s in negatives) / len(negatives)
        bacc = (tpr + tnr) / 2
        if best is None or bacc > best["balanced_accuracy"] + 1e-12:
            best = {"threshold": t, "balanced_accuracy": bacc, "tpr": tpr, "tnr": tnr}
    best.update(
        answer_queries=len(positives),
        abstain_queries=len(negatives),
        separable=max(negatives) < min(positives),
        max_abstain_score=max(negatives),
        min_answer_score=min(positives),
    )
    return best


def calibration_inputs(per_query: list[dict]) -> list[tuple[float, bool]]:
    scored = []
    for q in per_query:
        if q.get("top_score") is None:
            continue
        if q["expected_status"] == "ok" and q["relevant_eligible"] > 0:
            scored.append((q["top_score"], True))
        elif q["category"] == "irrelevant":
            scored.append((q["top_score"], False))
    return scored


# --- end to end ------------------------------------------------------------------------------


def _round(obj, digits: int = 6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def run_evaluation(
    split: str = "test",
    *,
    catalog_path: Path = DEFAULT_CATALOG,
    queries_path: Path = DEFAULT_QUERIES,
    meta_path: Path = DEFAULT_META,
    min_similarity: float | None = None,
    runs: int = 5,
    command: str = "",
) -> dict:
    """Build a fresh index, evaluate one split ('dev', 'test' or 'all'), return the full report."""
    if split not in ("dev", "test", "all"):
        raise EvaluationError("split must be dev, test or all")
    catalog = load_catalog(catalog_path)
    records, meta, queries_sha = load_eval_set(queries_path, meta_path, catalog=catalog)
    chosen = [r for r in records if split == "all" or r["split"] == split]

    with tempfile.TemporaryDirectory() as tmp:
        settings = IsolatedSettings(
            catalog_path=Path(catalog_path),
            artifact_dir=Path(tmp) / "index",
            explanation_mode="deterministic",
            embedding_backend="builtin",
            min_similarity=min_similarity,
        )
        embedder = LocalEmbedder()
        started = time.perf_counter()
        built = build_artifacts(catalog, embedder, settings.artifact_dir)
        build_seconds = time.perf_counter() - started
        manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
        deps = create_dependencies(settings, LocalEmbedder())
        per_query, operational = evaluate_records(chosen, deps, runs=runs)

    operational.update(
        catalog_build_seconds=build_seconds,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        request_count=len(chosen) * (runs + 1),
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "faiss": faiss.__version__,
        },
    )
    summary = aggregate(per_query)
    calibration = None
    scored = calibration_inputs(per_query)
    if split == "dev" and scored:
        try:
            calibration = calibrate_min_similarity(scored)
        except EvaluationError:
            calibration = None
    return _round(
        {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": command,
            "split": split,
            "configuration": {
                "settings": json.loads(settings.model_dump_json()),
                "retrieval_k": settings.retrieval_k,
                "max_recommendations": settings.max_recommendations,
                "explanation": "deterministic templates (no language model)",
                "embedder": manifest["embedding_model"],
                "embedder_revision": manifest["embedding_revision"],
                "document_format_version": DOCUMENT_FORMAT_VERSION,
                "catalog_file": str(Path(catalog_path).name),
                "catalog_sha256": catalog.sha256,
                "catalog_size": len(catalog),
                "index_version": manifest["catalog_sha256"][:12],
                "eval_set_sha256": queries_sha,
                "test_labels_sha256": meta["test_labels_sha256"],
                "random_baseline_draws": RANDOM_DRAWS,
                "latency_runs_per_query": runs,
            },
            "summary": summary,
            "operational": operational,
            "calibration": calibration,
            "per_query": per_query,
        }
    )


def render_summary(report: dict) -> str:
    """A short human-readable table of the headline numbers with denominators."""
    s, r = report["summary"], report["summary"]["retrieval"]

    def fmt(m: dict) -> str:
        return "n/a" if m["mean"] is None else f"{m['mean']:.3f} (n={m['n']})"

    def pct(m: dict) -> str:
        return (
            "n/a"
            if m["value"] is None
            else f"{m['value']:.3f} ({m['numerator']}/{m['denominator']})"
        )

    lines = [
        f"split={report['split']}  queries={s['queries']}  catalog={report['configuration']['catalog_size']} products",
        f"positive queries with relevant eligible products: {s['positive_queries_with_relevant_eligible']}",
        "",
        f"{'retrieval':<14}{'Recall@10':<18}{'MRR@3':<18}{'nDCG@3':<18}",
    ]
    for name in RANKERS:
        lines.append(
            f"{name:<14}{fmt(r[name]['recall_at_10']):<18}{fmt(r[name]['mrr_at_3']):<18}{fmt(r[name]['ndcg_at_3']):<18}"
        )
    lines.append(f"{'oracle (ceil)':<14}{fmt(r['oracle_recall_at_10'])}")
    sysm = s["system"]
    lines += ["", "end-to-end system:"]
    for key in (
        "status_accuracy", "interpretation_accuracy", "constraint_violation_rate",
        "answer_rate_on_positive_queries", "false_empty_rate", "no_match_accuracy",
        "clarification_accuracy", "abstention_on_irrelevant", "evidence_validity", "fallback_rate",
    ):  # fmt: skip
        lines.append(f"  {key:<34}{pct(sysm[key])}")
    lines.append(f"  {'mrr_at_3':<34}{fmt(sysm['mrr_at_3'])}")
    lines.append(f"  {'ndcg_at_3':<34}{fmt(sysm['ndcg_at_3'])}")
    op = report["operational"]
    latency = (
        f"latency warm p50={op['warm_p50_ms']:.2f} ms  p95={op['warm_p95_ms']:.2f} ms  "
        f"cold first request={op['cold_first_request_ms']:.2f} ms  "
        f"slowest stage={op.get('slowest_stage')}"
    )
    resources = (
        f"build={op['catalog_build_seconds']:.3f}s  peak RSS={op['peak_rss_mb']:.0f} MB  "
        f"requests={op['request_count']}"
    )
    lines += ["", latency, resources]
    return "\n".join(lines)


__all__ = ["EvaluationError", "calibrate_min_similarity", "render_summary", "run_evaluation"]
