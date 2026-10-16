from pathlib import Path

import pytest
from pydantic import ValidationError

from recommender.catalog import load_catalog
from recommender.query import parse_and_merge
from recommender.ranking import (
    build_evidence,
    evidence_id,
    matching_use_cases,
    select_candidates,
)
from recommender.retrieval import Candidate

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURE)


def cand(catalog, pid, score):
    return Candidate(pid, catalog.product_ids.index(pid), score)


def intent(text="laptop for coding and commuting", **kwargs):
    return parse_and_merge(text, **kwargs).intent


# --- selection -----------------------------------------------------------------------


def test_selection_orders_by_score_and_caps_at_requested_count(catalog):
    cands = [cand(catalog, p, s) for p, s in [("P001", 0.5), ("P004", 0.9), ("P006", 0.7)]]
    picked = select_candidates(cands, intent(), catalog)
    assert [c.product_id for c in picked] == ["P004", "P006", "P001"]
    capped = parse_and_merge("laptop", requested_count=2).intent
    assert [c.product_id for c in select_candidates(cands, capped, catalog)] == ["P004", "P006"]


def test_ties_are_broken_by_product_id(catalog):
    cands = [
        cand(catalog, "P006", 0.8),
        cand(catalog, "P001", 0.8 + 1e-9),
        cand(catalog, "P004", 0.8),
    ]
    picked = select_candidates(cands, intent(), catalog)
    assert [c.product_id for c in picked] == ["P001", "P004", "P006"]


def test_duplicates_are_collapsed_keeping_best_score(catalog):
    cands = [cand(catalog, "P001", 0.4), cand(catalog, "P001", 0.9), cand(catalog, "P004", 0.5)]
    picked = select_candidates(cands, intent(), catalog)
    assert [(c.product_id, c.score) for c in picked] == [("P001", 0.9), ("P004", 0.5)]


def test_ineligible_candidates_are_never_selected(catalog):
    # P002 costs 119900 and P007 is out of stock: retrieval cannot override eligibility.
    cands = [cand(catalog, "P002", 0.99), cand(catalog, "P007", 0.98), cand(catalog, "P001", 0.1)]
    picked = select_candidates(cands, intent("laptop under $900"), catalog)
    assert [c.product_id for c in picked] == ["P001"]


def test_fewer_than_requested_stay_fewer(catalog):
    picked = select_candidates([cand(catalog, "P004", 0.5)], intent(), catalog)
    assert len(picked) == 1


def test_selection_is_reproducible(catalog):
    cands = [cand(catalog, p, 0.5) for p in ("P004", "P001", "P006", "P003")]
    assert select_candidates(cands, intent(), catalog) == select_candidates(
        list(reversed(cands)), intent(), catalog
    )


def test_abstention_cutoff_is_optional_and_off_by_default(catalog):
    cands = [cand(catalog, "P001", 0.05)]
    assert select_candidates(cands, intent(), catalog)
    assert select_candidates(cands, intent(), catalog, min_score=0.2) == []


# --- use-case matching ---------------------------------------------------------------


def test_use_case_matching_uses_exact_words():
    uc = ["coding", "commuting", "office work", "3D work"]
    assert matching_use_cases(uc, "laptop for coding and travel") == ["coding"]
    assert matching_use_cases(uc, "office work and 3d work") == ["office work", "3D work"]
    assert matching_use_cases(uc, "I want to code") == []  # no stemming, no guessing
    assert matching_use_cases(uc, "work") == []


# --- evidence ------------------------------------------------------------------------


def evidence_by_field(catalog, pid, text):
    records = build_evidence([cand(catalog, pid, 0.5)], intent(text), catalog)
    return {r.field: r for r in records}


def test_evidence_ids_follow_the_documented_format(catalog):
    assert evidence_id("P001", "weight_kg") == "E_P001_WEIGHT"
    ev = evidence_by_field(catalog, "P001", "laptop")
    assert ev["weight_kg"].evidence_id == "E_P001_WEIGHT"
    assert ev["weight_kg"].value == 1.3
    assert ev["weight_kg"].product_id == "P001"


def test_constraint_reasons_require_an_active_constraint(catalog):
    inactive = evidence_by_field(catalog, "P001", "laptop for coding")
    assert inactive["weight_kg"].allowed_reason_codes == ("catalog_fact",)
    assert inactive["ram_gb"].allowed_reason_codes == ("catalog_fact",)
    assert inactive["price_cents"].allowed_reason_codes == ("catalog_fact",)

    active = evidence_by_field(
        catalog, "P001", "laptop for coding under $900, at least 16 GB RAM, at most 1.5 kg"
    )
    assert active["weight_kg"].allowed_reason_codes == ("meets_weight_limit", "catalog_fact")
    assert active["ram_gb"].allowed_reason_codes == ("meets_ram_minimum", "catalog_fact")
    assert active["price_cents"].allowed_reason_codes == ("meets_budget", "catalog_fact")
    assert active["in_stock"].allowed_reason_codes == ("in_stock", "catalog_fact")


def test_use_case_reason_only_when_the_request_names_it(catalog):
    hit = evidence_by_field(catalog, "P001", "laptop for coding")["use_cases"]
    assert hit.allowed_reason_codes == ("use_case_match", "catalog_fact")
    miss = evidence_by_field(catalog, "P001", "laptop for gaming")["use_cases"]
    assert miss.allowed_reason_codes == ("catalog_fact",)


def test_unknown_spec_has_no_allowed_reason_codes(catalog):
    ev = evidence_by_field(catalog, "P008", "laptop for coding")
    assert ev["weight_kg"].value is None
    assert ev["weight_kg"].allowed_reason_codes == ()


def test_evidence_values_exactly_match_the_catalog(catalog):
    for pid in ("P001", "P004", "P008"):
        product = catalog.get(pid)
        ev = evidence_by_field(catalog, pid, "laptop")
        assert ev["price_cents"].value == product.price_cents
        assert ev["ram_gb"].value == product.ram_gb
        assert ev["gpu_type"].value == product.gpu_type
        assert ev["use_cases"].value == product.use_cases


def test_evidence_is_immutable(catalog):
    record = evidence_by_field(catalog, "P001", "laptop")["price_cents"]
    with pytest.raises(ValidationError):
        record.value = 1


def test_evidence_covers_every_selected_product_in_order(catalog):
    selected = [cand(catalog, "P004", 0.9), cand(catalog, "P001", 0.8)]
    records = build_evidence(selected, intent(), catalog)
    assert [r.product_id for r in records] == ["P004"] * 7 + ["P001"] * 7
