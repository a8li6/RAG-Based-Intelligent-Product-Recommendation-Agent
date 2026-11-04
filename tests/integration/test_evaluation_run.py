"""End-to-end evaluation runs on the real demo catalog and golden set."""

import json
from pathlib import Path

import pytest

from recommender.evaluation import EvaluationError, render_summary, run_evaluation

ROOT = Path(__file__).resolve().parents[2]


def strip_timing(report: dict) -> dict:
    """Everything except wall-clock and process measurements, which legitimately vary."""
    return {k: report[k] for k in ("split", "summary", "per_query", "calibration")} | {
        "configuration": {k: v for k, v in report["configuration"].items() if k != "settings"}
    }


@pytest.fixture(scope="module")
def dev():
    return run_evaluation("dev", runs=1)


def test_dev_split_has_twenty_queries_and_a_complete_report(dev):
    assert dev["split"] == "dev"
    assert dev["summary"]["queries"] == len(dev["per_query"]) == 20
    assert {q["split"] for q in dev["per_query"]} == {"dev"}
    cfg = dev["configuration"]
    assert cfg["catalog_size"] == 60
    assert cfg["embedder"] == "builtin-hashed-tfidf-v1"
    assert cfg["embedder_revision"].startswith("idf-")
    assert len(cfg["catalog_sha256"]) == len(cfg["eval_set_sha256"]) == 64
    assert cfg["settings"]["explanation_mode"] == "deterministic"  # the full run configuration
    op = dev["operational"]
    assert op["warm_p95_ms"] >= op["warm_p50_ms"] > 0
    assert op["slowest_stage"] in {"parse", "filter", "retrieve", "select", "explain", "render"}
    assert op["hardware"]["python"] and op["peak_rss_mb"] > 0
    json.dumps(dev)  # the report must be serializable as-is


def test_per_query_outputs_are_published_with_their_rankings(dev):
    positive = next(q for q in dev["per_query"] if q["expected_status"] == "ok")
    assert set(positive["rankings"]) == {"builtin", "keyword", "random", "post_filter"}
    assert set(positive["retrieval"]) == {"builtin", "keyword", "random", "post_filter"}
    assert positive["returned"] and positive["status"] == "ok"


def test_hard_constraints_are_never_violated(dev):
    assert all(q["violations"] == [] for q in dev["per_query"])
    rate = dev["summary"]["system"]["constraint_violation_rate"]
    assert rate["numerator"] == 0 and rate["denominator"] > 0


def test_every_displayed_claim_survives_the_independent_audit(dev):
    rate = dev["summary"]["system"]["evidence_validity"]
    assert rate["denominator"] > 0
    assert rate["numerator"] == rate["denominator"]


def test_constraints_are_interpreted_correctly_on_dev(dev):
    rate = dev["summary"]["system"]["interpretation_accuracy"]
    assert rate["numerator"] == rate["denominator"]


def test_unknown_required_specs_are_excluded(dev):
    u1 = next(q for q in dev["per_query"] if q["query_id"] == "U1")
    assert "P045" not in u1["returned"]  # relevant, but its weight is unknown


def test_filtering_first_beats_filtering_after_a_global_top_k(dev):
    r = dev["summary"]["retrieval"]
    assert r["builtin"]["recall_at_10"]["mean"] > r["post_filter"]["recall_at_10"]["mean"]
    assert r["builtin"]["recall_at_10"]["mean"] > 1.5 * r["random"]["recall_at_10"]["mean"]


def test_the_recall_ceiling_is_reported_next_to_recall(dev):
    r = dev["summary"]["retrieval"]
    assert r["builtin"]["recall_at_10"]["mean"] <= r["oracle_recall_at_10"]["mean"] + 1e-9


def test_denominators_exclude_queries_without_relevant_items(dev):
    s = dev["summary"]
    assert (
        s["positive_queries_with_relevant_eligible"]
        == s["retrieval"]["builtin"]["recall_at_10"]["n"]
    )
    assert s["positive_queries_with_relevant_eligible"] < s["queries"]


def test_the_evaluation_is_reproducible(dev):
    again = run_evaluation("dev", runs=1)
    assert strip_timing(again) == strip_timing(dev)


def test_calibration_on_dev_separates_irrelevant_from_answerable_queries(dev):
    cal = dev["calibration"]
    assert cal["answer_queries"] > 0 and cal["abstain_queries"] >= 2
    assert cal["separable"] is True
    assert cal["max_abstain_score"] < cal["threshold"] <= cal["min_answer_score"]


def test_the_calibrated_cutoff_makes_irrelevant_queries_abstain(dev):
    threshold = dev["calibration"]["threshold"]
    before = dev["summary"]["system"]["abstention_on_irrelevant"]
    after = run_evaluation("dev", runs=1, min_similarity=threshold)["summary"]["system"]
    assert before["numerator"] == 0  # no cutoff: it answers nonsense
    assert (
        after["abstention_on_irrelevant"]["numerator"]
        == after["abstention_on_irrelevant"]["denominator"]
    )
    assert after["false_empty_rate"]["numerator"] == 0
    assert after["answer_rate_on_positive_queries"]["value"] == 1.0  # and answers stay answered


def test_split_all_covers_every_query():
    report = run_evaluation("all", runs=1)
    assert report["summary"]["queries"] == 60
    assert report["calibration"] is None  # calibration is a development-only activity


def test_summary_text_shows_every_retrieval_method_with_denominators(dev):
    text = render_summary(dev)
    for name in ("builtin", "keyword", "random", "post_filter", "oracle"):
        assert name in text
    assert "n=16" in text and "constraint_violation_rate" in text


def test_unknown_split_is_rejected():
    with pytest.raises(EvaluationError):
        run_evaluation("validation")


def test_a_changed_catalog_stops_the_evaluation(tmp_path):
    lines = (ROOT / "data" / "demo" / "products.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[10])
    record["price_cents"] += 100
    lines[10] = json.dumps(record)
    changed = tmp_path / "products.jsonl"
    changed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="label drift"):
        run_evaluation("dev", catalog_path=changed, runs=1)
