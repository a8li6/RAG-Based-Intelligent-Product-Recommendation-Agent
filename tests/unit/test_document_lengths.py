import json
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.documents import (
    DocumentTooLongError,
    catalog_documents,
    check_document_lengths,
)
from recommender.embeddings import PASSAGE_PREFIX, E5Embedder
from recommender.index import MANIFEST_FILE, build_artifacts

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURE)


def words(text):
    return len(text.split())


def test_stats_are_reported_for_valid_documents(catalog):
    stats = check_document_lengths(catalog, words)
    assert stats["count"] == 8
    assert 0 < stats["mean"] <= stats["max"] <= stats["limit"] == 512


def test_prefix_is_counted_as_embedded(catalog):
    seen = []
    check_document_lengths(catalog, lambda text: seen.append(text) or 1)
    assert all(text.startswith(PASSAGE_PREFIX) for text in seen)
    assert len(seen) == 8


def test_overlong_documents_are_rejected_not_truncated(catalog):
    with pytest.raises(DocumentTooLongError) as err:
        check_document_lengths(catalog, words, limit=20)
    assert set(err.value.too_long) == set(catalog.product_ids)
    assert "P001" in str(err.value)
    assert "truncation" in str(err.value)


def test_only_the_overlong_product_is_listed(catalog):
    lengths = {doc: (100 if "Atlas Air" in doc else 10) for doc in catalog_documents(catalog)}

    def counter(text):
        return lengths[text.removeprefix(PASSAGE_PREFIX)]

    with pytest.raises(DocumentTooLongError) as err:
        check_document_lengths(catalog, counter, limit=50)
    assert err.value.too_long == {"P001": 100}


def test_exact_limit_is_allowed(catalog):
    assert check_document_lengths(catalog, lambda text: 512)["max"] == 512
    with pytest.raises(DocumentTooLongError):
        check_document_lengths(catalog, lambda text: 513)


def test_build_records_token_stats_in_the_manifest(tmp_path, catalog, embedder):
    built = build_artifacts(catalog, embedder, tmp_path / "catalog-v1")
    stats = json.loads((built / MANIFEST_FILE).read_text())["document_tokens"]
    assert stats["count"] == 8
    assert stats["max"] <= stats["limit"]


def test_build_fails_fast_and_leaves_nothing_when_a_document_is_too_long(
    tmp_path, catalog, embedder
):
    embedder.max_tokens = 20
    with pytest.raises(DocumentTooLongError):
        build_artifacts(catalog, embedder, tmp_path / "catalog-v1")
    assert list(tmp_path.iterdir()) == []


def test_embedder_without_a_tokenizer_records_no_stats(tmp_path, catalog, embedder):
    class NoTokenizer:
        model_id = embedder.model_id
        revision = embedder.revision
        encode_products = staticmethod(embedder.encode_products)
        encode_query = staticmethod(embedder.encode_query)

    built = build_artifacts(catalog, NoTokenizer(), tmp_path / "catalog-v1")
    assert json.loads((built / MANIFEST_FILE).read_text())["document_tokens"] is None


def test_real_embedder_exposes_token_counting_without_loading_a_model():
    real = E5Embedder()
    assert callable(real.count_tokens)
    assert real._model is None
