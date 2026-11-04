import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.schemas import Constraints

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "data" / "eval" / "queries.jsonl"
META = ROOT / "data" / "eval" / "meta.json"
CATALOG = ROOT / "data" / "demo" / "products.jsonl"


@pytest.fixture(scope="module")
def rows():
    return [json.loads(x) for x in QUERIES.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def meta():
    return json.loads(META.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(CATALOG)


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "build_eval_set", ROOT / "scripts" / "build_eval_set.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_files_match_the_generator(generator):
    records, meta = generator.build()
    queries, meta_text = generator.render(records, meta)
    assert QUERIES.read_text(encoding="utf-8") == queries
    assert META.read_text(encoding="utf-8") == meta_text


def test_sizes_and_split_counts(rows, meta):
    assert len(rows) == 60 == meta["query_count"]
    assert Counter(r["split"] for r in rows) == {"dev": 20, "test": 40}
    assert meta["splits"] == {"dev": 20, "test": 40}


def test_catalog_hash_is_stored_so_label_drift_is_detectable(meta, catalog):
    assert meta["catalog_sha256"] == catalog.sha256


def test_ids_and_texts_are_unique(rows):
    assert len({r["query_id"] for r in rows}) == 60
    assert len({r["query"] for r in rows}) == 60


def test_paraphrase_groups_never_straddle_splits(rows):
    splits_by_group = defaultdict(set)
    for r in rows:
        splits_by_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_by_group.values())
    paraphrase_groups = Counter(r["group"] for r in rows if r["category"] == "paraphrase")
    assert sorted(paraphrase_groups.values()) == [3, 3, 3, 3]


def test_every_failure_mode_from_the_design_is_covered(rows):
    categories = {r["category"] for r in rows}
    assert categories >= {
        "direct", "paraphrase", "constraints", "model-name", "unknown-spec", "no-eligible",
        "ambiguous", "contradictory", "irrelevant", "negation", "injection",
    }  # fmt: skip


def test_dev_and_test_both_contain_irrelevant_queries(rows):
    for split in ("dev", "test"):
        assert sum(r["category"] == "irrelevant" and r["split"] == split for r in rows) >= 2


def test_eligibility_is_recomputed_from_the_stored_constraints(rows, catalog):
    for r in rows:
        constraints = Constraints(**r["constraints"])
        expected = [catalog.products[i].product_id for i in catalog.eligible_rows(constraints)]
        assert r["eligible_ids"] == expected, r["query_id"]


def test_expected_status_is_consistent_with_eligibility(rows):
    for r in rows:
        if r["expected_status"] == "no_match":
            assert r["eligible_ids"] == [], r["query_id"]
        if r["expected_status"] == "ok":
            assert r["eligible_ids"] and r["relevance"], r["query_id"]
        if r["expected_status"] == "needs_clarification":
            assert r["category"] in {"ambiguous", "contradictory", "irrelevant"}, r["query_id"]


def test_relevance_labels_reference_real_products_with_valid_grades(rows, catalog):
    ids = set(catalog.product_ids)
    for r in rows:
        assert set(r["relevance"]) <= ids, r["query_id"]
        assert set(r["relevance"].values()) <= {1, 2}, r["query_id"]


def test_irrelevant_and_unanswerable_queries_have_no_relevant_items(rows):
    for r in rows:
        if r["category"] in {"irrelevant", "ambiguous", "contradictory", "no-eligible"}:
            assert r["relevance"] == {}, r["query_id"]


def test_unknown_spec_queries_label_an_excluded_product_as_relevant(rows):
    # The system must not return it: relevant but not eligible.
    u1 = next(r for r in rows if r["query_id"] == "U1")
    assert "P045" in u1["relevance"] and "P045" not in u1["eligible_ids"]
    u2 = next(r for r in rows if r["query_id"] == "U2")
    assert "P060" in u2["relevance"] and "P060" not in u2["eligible_ids"]


def test_negation_labels_veto_the_negated_tag(rows, catalog):
    g1 = next(r for r in rows if r["query_id"] == "G1")
    for pid in g1["relevance"]:
        assert "gaming" not in catalog.get(pid).use_cases
    assert "P054" not in g1["relevance"]  # tagged study and gaming
    g3 = next(r for r in rows if r["query_id"] == "G3")
    assert all(catalog.get(p).gpu_type == "integrated" for p in g3["relevance"])


def test_model_name_queries_label_the_exact_product_strong(rows, catalog):
    for r in rows:
        if r["category"] == "model-name":
            strong = [p for p, g in r["relevance"].items() if g == 2]
            assert [catalog.get(p).title for p in strong] == [r["query"]]


def test_test_labels_are_frozen_by_digest(rows, meta, generator):
    import hashlib

    frozen = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["query_id"])
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert digest == meta["test_labels_sha256"]


def test_the_set_states_its_limits(meta):
    assert "Single author" in meta["labeling"]
    assert "frozen" in meta["split_policy"]
