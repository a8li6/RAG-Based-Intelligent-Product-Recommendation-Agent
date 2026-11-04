import json
import shutil
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.documents import catalog_documents
from recommender.evaluation import (
    Bm25,
    EvaluationError,
    aggregate,
    audit_item,
    calibrate_min_similarity,
    calibration_inputs,
    frozen_test_labels_digest,
    load_eval_set,
)
from recommender.schemas import RecommendationItem

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "demo" / "products.jsonl"
QUERIES = ROOT / "data" / "eval" / "queries.jsonl"
META = ROOT / "data" / "eval" / "meta.json"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(CATALOG)


# --- integrity ---------------------------------------------------------------------------


def test_the_committed_set_loads_and_matches_the_catalog(catalog):
    records, meta, sha = load_eval_set(QUERIES, META, catalog=catalog)
    assert len(records) == 60 and len(sha) == 64
    assert frozen_test_labels_digest(records) == meta["test_labels_sha256"]


def test_label_drift_is_detected_when_the_catalog_changes(catalog, tmp_path):
    lines = CATALOG.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["price_cents"] += 1
    lines[0] = json.dumps(record)
    changed = tmp_path / "changed.jsonl"
    changed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="label drift"):
        load_eval_set(QUERIES, META, catalog=load_catalog(changed))


def test_changed_test_labels_are_detected(catalog, tmp_path):
    rows = [json.loads(x) for x in QUERIES.read_text(encoding="utf-8").splitlines()]
    target = next(r for r in rows if r["split"] == "test" and r["relevance"])
    target["relevance"][next(iter(target["relevance"]))] = (
        1 if target["relevance"][next(iter(target["relevance"]))] == 2 else 2
    )
    edited = tmp_path / "queries.jsonl"
    edited.write_text(
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    shutil.copy(META, tmp_path / "meta.json")
    with pytest.raises(EvaluationError, match="frozen test labels"):
        load_eval_set(edited, tmp_path / "meta.json", catalog=catalog)


def test_changing_a_dev_label_does_not_trip_the_test_freeze(catalog, tmp_path):
    rows = [json.loads(x) for x in QUERIES.read_text(encoding="utf-8").splitlines()]
    dev = next(r for r in rows if r["split"] == "dev" and r["relevance"])
    dev["notes"] = "edited"
    edited = tmp_path / "queries.jsonl"
    edited.write_text(
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    assert load_eval_set(edited, META, catalog=catalog)


# --- keyword baseline --------------------------------------------------------------------


def test_bm25_ranks_the_exact_model_name_first(catalog):
    bm25 = Bm25(catalog_documents(catalog))
    rows = list(range(len(catalog)))
    ranked = bm25.rank("Atlas Air 14", rows, 3)
    assert catalog.products[ranked[0]].title == "Atlas Air 14"


def test_bm25_restricts_to_the_given_rows_and_breaks_ties_by_row(catalog):
    bm25 = Bm25(catalog_documents(catalog))
    assert bm25.rank("zzzz qqqq", [5, 2, 9], 3) == [2, 5, 9]  # all zero scores: row order
    assert set(bm25.rank("gaming", [0, 1, 2], 10)) == {0, 1, 2}


# --- evidence audit ----------------------------------------------------------------------


def item_for(catalog, pid="P001", **overrides):
    p = catalog.get(pid)
    money = f"{p.price_cents // 100}.{p.price_cents % 100:02d}"
    default_reasons = [
        f"Priced at {money} USD, within your maximum of 900.00 USD.",
        f"Has {p.ram_gb} GB RAM, meeting your minimum of 16 GB.",
        "Marked in stock in this catalog snapshot.",
        "The catalog lists use cases related to your request: coding, commuting.",
        "The catalog lists GPU: integrated.",
    ]
    if p.weight_kg is not None:
        default_reasons.insert(2, f"Weighs {p.weight_kg:.2f} kg, within your limit of 1.5 kg.")
    base = {
        "product_id": pid,
        "title": p.title,
        "price_cents": p.price_cents,
        "currency": p.currency,
        "in_stock": p.in_stock,
        "source_url": p.source_url,
        "attributes": {
            "ram_gb": p.ram_gb,
            "storage_gb": p.storage_gb,
            "weight_kg": p.weight_kg,
            "gpu_type": p.gpu_type,
            "use_cases": list(p.use_cases),
        },
        "reasons": default_reasons,
        "evidence_refs": [f"{pid}.price_cents", f"{pid}.ram_gb"],
        "unknown_fields": [],
    }
    base.update(overrides)
    return RecommendationItem(**base)


def test_audit_accepts_a_faithful_item(catalog):
    checked, valid = audit_item(item_for(catalog), catalog)
    assert checked == valid == 10 + 2 + 6


@pytest.mark.parametrize(
    "override",
    [
        {"price_cents": 1},
        {"title": "Other"},
        {"in_stock": False},
        {"source_url": "https://evil.example/x"},
        {"evidence_refs": ["P002.ram_gb"]},
        {"evidence_refs": ["P001.discount"]},
        {"reasons": ["Priced at 1.00 USD, within your maximum of 900.00 USD."]},
        {"reasons": ["Has 64 GB RAM, meeting your minimum of 16 GB."]},
        {"reasons": ["Weighs 0.50 kg, within your limit of 1.5 kg."]},
        {"reasons": ["The catalog lists use cases related to your request: gaming."]},
        {"reasons": ["The catalog lists GPU: dedicated."]},
        {"reasons": ["This laptop is the fastest on the market."]},
    ],
)
def test_audit_catches_every_kind_of_unsupported_claim(catalog, override):
    checked, valid = audit_item(item_for(catalog, **override), catalog)
    assert valid < checked


def test_audit_flags_wrong_attributes(catalog):
    item = item_for(catalog)
    item.attributes["ram_gb"] = 128
    checked, valid = audit_item(item, catalog)
    assert valid == checked - 1


def test_unknown_values_are_audited_as_unknown(catalog):
    item = item_for(catalog, "P008", reasons=[], evidence_refs=[])
    checked, valid = audit_item(item, catalog)
    assert checked == valid == 10
    assert item.attributes["weight_kg"] is None


# --- calibration -------------------------------------------------------------------------


def test_calibration_picks_a_cutoff_inside_the_gap_when_separable():
    result = calibrate_min_similarity([(0.30, True), (0.45, True), (0.02, False), (0.05, False)])
    assert result["separable"] is True
    assert 0.05 < result["threshold"] <= 0.30
    assert result["balanced_accuracy"] == 1.0
    assert (result["answer_queries"], result["abstain_queries"]) == (2, 2)


def test_calibration_maximizes_balanced_accuracy_when_not_separable():
    scored = [(0.5, True), (0.4, True), (0.1, True), (0.3, False), (0.05, False)]
    result = calibrate_min_similarity(scored)
    assert result["separable"] is False
    assert 0.5 < result["balanced_accuracy"] < 1.0 or result["balanced_accuracy"] == pytest.approx(
        5 / 6
    )
    assert result["threshold"] > 0.05


@pytest.mark.parametrize("scored", [[], [(0.3, True)], [(0.1, False), (0.2, False)]])
def test_calibration_needs_both_classes(scored):
    with pytest.raises(EvaluationError):
        calibrate_min_similarity(scored)


def test_calibration_inputs_use_only_answerable_and_irrelevant_queries():
    rows = [
        {"top_score": 0.4, "expected_status": "ok", "relevant_eligible": 3, "category": "direct"},
        {
            "top_score": 0.01,
            "expected_status": "needs_clarification",
            "relevant_eligible": 0,
            "category": "irrelevant",
        },
        {
            "top_score": None,
            "expected_status": "no_match",
            "relevant_eligible": 0,
            "category": "no-eligible",
        },
        {
            "top_score": 0.9,
            "expected_status": "needs_clarification",
            "relevant_eligible": 0,
            "category": "ambiguous",
        },
    ]
    assert calibration_inputs(rows) == [(0.4, True), (0.01, False)]


# --- aggregation and denominators ----------------------------------------------------------


def q(**kw):
    base = dict(
        query_id="Q", split="dev", category="direct", expected_status="ok", status="ok", returned=["P001"],
        violations=[], interpretation_ok=True, claims_checked=10, claims_valid=10, explanation_mode="deterministic",
        fallback_reason="llm_disabled", relevant_eligible=2, system_mrr_at_3=1.0, system_ndcg_at_3=0.8,
        lexicon_hit_expected=True, oracle_recall_at_10=1.0,
        retrieval={name: {"recall_at_10": 0.5, "mrr_at_3": 1.0, "ndcg_at_3": 0.7} for name in ("builtin", "keyword", "random", "post_filter")},
    )  # fmt: skip
    base.update(kw)
    return base


def test_aggregate_reports_denominators_and_na_for_zero():
    rows = [
        q(),
        q(
            query_id="R",
            status="needs_clarification",
            returned=[],
            expected_status="needs_clarification",
            category="ambiguous",
            relevant_eligible=0,
            claims_checked=0,
            claims_valid=0,
        ),
    ]
    s = aggregate(rows)["system"]
    assert s["constraint_violation_rate"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    assert s["status_accuracy"]["denominator"] == 2
    assert s["clarification_accuracy"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert s["no_match_accuracy"]["value"] is None  # no no_match queries: N/A, not perfect
    only_clarified = aggregate([rows[1]])["system"]
    assert only_clarified["constraint_violation_rate"]["value"] is None  # nothing returned: N/A


def test_aggregate_counts_violations_per_returned_product():
    rows = [q(returned=["P001", "P002", "P003"], violations=["P002"])]
    assert aggregate(rows)["system"]["constraint_violation_rate"]["value"] == pytest.approx(1 / 3)


def test_queries_without_relevant_items_are_excluded_and_counted():
    rows = [
        q(),
        q(
            query_id="Z",
            relevant_eligible=0,
            retrieval={
                n: {"recall_at_10": None, "mrr_at_3": None, "ndcg_at_3": None}
                for n in ("builtin", "keyword", "random", "post_filter")
            },
        ),
    ]
    summary = aggregate(rows)
    assert summary["positive_queries_with_relevant_eligible"] == 1
    assert summary["retrieval"]["builtin"]["recall_at_10"]["n"] == 1
