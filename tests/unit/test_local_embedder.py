from pathlib import Path

import numpy as np
import pytest

from recommender.catalog import load_catalog
from recommender.documents import catalog_documents
from recommender.embeddings import EmbeddingError, validate_vectors
from recommender.local_embedder import (
    DIM,
    STATE_JSON,
    STATE_NPY,
    LocalEmbedder,
    features,
    stem,
    tokenize,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURE)


@pytest.fixture(scope="module")
def fitted(catalog):
    return LocalEmbedder().fit(catalog_documents(catalog))


def top_ids(embedder, catalog, query, k=2):
    vectors = embedder.encode_products(catalog_documents(catalog))
    scores = vectors @ embedder.encode_query(query)[0]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [catalog.products[i].product_id for i in order[:k]]


# --- text processing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [("coding", "cod"), ("code", "cod"), ("coder", "cod"), ("commuting", "commut"),
     ("commuter", "commut"), ("commute", "commut"), ("development", "develop"),
     ("developer", "develop"), ("laptops", "laptop"), ("gpu", "gpu"), ("ram", "ram")],
)  # fmt: skip
def test_stemmer(word, expected):
    assert stem(word) == expected


def test_stopwords_are_removed_and_text_is_lowercased():
    assert tokenize("I need a Laptop for the CODING") == ["laptop", "coding"]


def test_synonyms_share_a_concept_feature():
    assert "concept_code" in features("programming") and "concept_code" in features("coding")
    assert "concept_travel" in features("portable") and "concept_travel" in features("commuting")
    assert "concept_code" not in features("gaming")


# --- vector contract -----------------------------------------------------------------


def test_vectors_satisfy_the_embedding_contract(fitted, catalog):
    products = fitted.encode_products(catalog_documents(catalog))
    assert products.shape == (8, DIM)
    validate_vectors(products, expected_rows=8, expected_dim=DIM)
    validate_vectors(fitted.encode_query("portable laptop"), expected_rows=1, expected_dim=DIM)


def test_encoding_is_deterministic(catalog):
    docs = catalog_documents(catalog)
    a = LocalEmbedder().fit(docs)
    b = LocalEmbedder().fit(docs)
    assert a.revision == b.revision
    np.testing.assert_array_equal(a.encode_products(docs), b.encode_products(docs))
    np.testing.assert_array_equal(a.encode_query("coding"), b.encode_query("coding"))


@pytest.mark.parametrize("text", ["", "   ", "the a of", "!!! ???", "مرحبا"])
def test_texts_without_features_still_give_a_valid_unit_vector(fitted, text):
    validate_vectors(fitted.encode_query(text))


def test_revision_is_unfitted_until_fitted_and_changes_with_the_corpus(catalog):
    assert LocalEmbedder().revision == "unfitted"
    docs = catalog_documents(catalog)
    assert LocalEmbedder().fit(docs).revision != LocalEmbedder().fit(docs[:4]).revision


def test_fit_rejects_an_empty_corpus():
    with pytest.raises(EmbeddingError):
        LocalEmbedder().fit([])


# --- retrieval behavior on the reference fixture --------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_any_of"),
    [
        ("laptop for gaming and 3D work", {"P002", "P005"}),
        ("study and web browsing", {"P003"}),
        ("portable laptop for travel", {"P007", "P004"}),
        ("laptop for programming and development", {"P001", "P004", "P006", "P008", "P002"}),
        ("content creation video editing", {"P005", "P002"}),
        ("home office laptop", {"P006", "P001", "P007", "P008"}),
    ],
)
def test_top_result_matches_the_catalog_use_cases(fitted, catalog, query, expected_any_of):
    assert top_ids(fitted, catalog, query, k=1)[0] in expected_any_of


def test_synonym_query_finds_the_same_products_as_the_catalog_word(fitted, catalog):
    # The catalog says "coding"; the query says "programming". They share a concept token.
    assert set(top_ids(fitted, catalog, "programming", 3)) == set(
        top_ids(fitted, catalog, "coding", 3)
    )


def test_misspelling_still_prefers_the_right_product(fitted, catalog):
    assert top_ids(fitted, catalog, "comuting laptop", k=3)[0] in {"P004", "P001", "P007"}


def test_unrelated_query_scores_low(fitted, catalog):
    vectors = fitted.encode_products(catalog_documents(catalog))
    related = (vectors @ fitted.encode_query("laptop for coding")[0]).max()
    unrelated = (vectors @ fitted.encode_query("banana smoothie recipe")[0]).max()
    assert unrelated < related


# --- persisted state -----------------------------------------------------------------


def test_state_round_trip_reproduces_identical_vectors(fitted, catalog):
    restored = LocalEmbedder()
    restored.load_state(fitted.state_files())
    assert restored.revision == fitted.revision
    docs = catalog_documents(catalog)
    np.testing.assert_array_equal(restored.encode_products(docs), fitted.encode_products(docs))
    np.testing.assert_array_equal(restored.encode_query("coding"), fitted.encode_query("coding"))


def test_state_is_loaded_with_allow_pickle_false(fitted):
    assert set(fitted.state_files()) == {STATE_JSON, STATE_NPY}


@pytest.mark.parametrize("mutate", ["missing_json", "missing_npy", "bad_model", "bad_dim", "nan"])
def test_bad_state_is_rejected(fitted, mutate):
    import io
    import json

    files = dict(fitted.state_files())
    if mutate == "missing_json":
        del files[STATE_JSON]
    elif mutate == "missing_npy":
        del files[STATE_NPY]
    elif mutate == "bad_model":
        files[STATE_JSON] = json.dumps({"model_id": "other", "dim": DIM}).encode()
    elif mutate == "bad_dim":
        buf = io.BytesIO()
        np.save(buf, np.ones(10, dtype=np.float32), allow_pickle=False)
        files[STATE_NPY] = buf.getvalue()
    elif mutate == "nan":
        buf = io.BytesIO()
        np.save(buf, np.full(DIM, np.nan, dtype=np.float32), allow_pickle=False)
        files[STATE_NPY] = buf.getvalue()
    with pytest.raises(EmbeddingError):
        LocalEmbedder().load_state(files)


def test_unfitted_state_cannot_be_exported():
    with pytest.raises(EmbeddingError):
        LocalEmbedder().state_files()
