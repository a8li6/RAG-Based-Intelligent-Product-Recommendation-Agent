from pathlib import Path

import numpy as np
import pytest

from recommender.catalog import load_catalog
from recommender.index import ArtifactError, build_artifacts, load_artifacts
from recommender.retrieval import Retriever, search_eligible
from recommender.schemas import Constraints

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


def unit_rows(n, dim=8, seed=0):
    arr = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_inner_product_equals_cosine_for_unit_vectors():
    vectors = unit_rows(5)
    query = vectors[2:3]
    hits = search_eligible(query, vectors, list(range(5)), 5)
    assert hits[0][0] == 2
    for row, score in hits:
        expected = float(vectors[row] @ query[0] / (np.linalg.norm(vectors[row]) * 1.0))
        assert score == pytest.approx(expected, abs=1e-5)


def test_results_map_back_to_original_rows_and_are_sorted():
    vectors = unit_rows(10)
    eligible = [7, 2, 9]
    hits = search_eligible(vectors[9:10], vectors, eligible, 3)
    assert {row for row, _ in hits} == {2, 7, 9}
    assert hits[0][0] == 9
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_only_eligible_item_below_global_top_k_is_still_found():
    # 12 rows; the query matches rows 0-11 in order, so row 11 is the worst global match.
    dim = 12
    vectors = np.eye(dim, dtype=np.float32)
    query = np.array([[12 - i for i in range(dim)]], dtype=np.float32)
    query /= np.linalg.norm(query)
    import faiss

    full = faiss.IndexFlatIP(dim)
    full.add(vectors)
    _, global_top10 = full.search(query, 10)
    assert 11 not in global_top10[0]  # post-filtering a global top-10 would lose it
    hits = search_eligible(query, vectors, [11], 10)
    assert [row for row, _ in hits] == [11]


def test_empty_subset_returns_nothing():
    assert search_eligible(unit_rows(1)[0], unit_rows(5), [], 3) == []


def test_k_larger_than_eligible_count_is_capped():
    vectors = unit_rows(6)
    assert len(search_eligible(vectors[0], vectors, [1, 3], 10)) == 2


def test_duplicate_eligible_rows_do_not_duplicate_results():
    vectors = unit_rows(6)
    hits = search_eligible(vectors[0], vectors, [1, 1, 3, 3], 10)
    assert sorted(row for row, _ in hits) == [1, 3]


@pytest.mark.parametrize("rows", [[-1], [6], [0, 99]])
def test_out_of_range_rows_are_rejected(rows):
    with pytest.raises(ValueError, match="out of range"):
        search_eligible(unit_rows(1)[0], unit_rows(6), rows, 3)


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        search_eligible(unit_rows(1)[0], unit_rows(3), [0], 0)


def test_wrong_query_dimension_is_rejected():
    with pytest.raises(ValueError, match="dimension"):
        search_eligible(np.ones(3, dtype=np.float32), unit_rows(3), [0], 1)


@pytest.fixture
def retriever(tmp_path, embedder):
    built = build_artifacts(load_catalog(FIXTURE), embedder, tmp_path / "catalog-v1")
    return Retriever(load_artifacts(built), embedder, retrieval_k=10)


def test_fixture_search_returns_exactly_the_eligible_ids(retriever):
    constraints = Constraints(max_price_cents=90000, min_ram_gb=16, max_weight_kg=1.5)
    rows = retriever.eligible_rows(constraints)
    for query in ("portable laptop for coding", "gaming 3D rendering", "zzz unrelated words"):
        ids = {c.product_id for c in retriever.search(query, rows)}
        assert ids == {"P001", "P004"}, query


def test_fixture_no_match_never_returns_candidates(retriever):
    rows = retriever.eligible_rows(Constraints(max_price_cents=30000, min_ram_gb=16))
    assert rows == []
    assert retriever.search("laptop for coding", rows) == []


def test_search_is_reproducible(retriever):
    rows = retriever.eligible_rows(Constraints())
    first = retriever.search("laptop for coding and commuting", rows)
    second = retriever.search("laptop for coding and commuting", rows)
    assert first == second
    assert len(first) == 7  # k=10 capped at the seven in-stock products


def test_retriever_rejects_mismatched_embedder(tmp_path, embedder):
    built = build_artifacts(load_catalog(FIXTURE), embedder, tmp_path / "catalog-v1")
    artifacts = load_artifacts(built)
    embedder.revision = "different"
    with pytest.raises(ArtifactError, match="rebuild"):
        Retriever(artifacts, embedder)
