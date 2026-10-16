import numpy as np
import pytest

from recommender.embeddings import E5Embedder, EmbeddingError, validate_vectors


def unit(rows, dim=4):
    arr = np.random.default_rng(0).normal(size=(rows, dim)).astype(np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_valid_vectors_pass():
    vectors = unit(3)
    assert validate_vectors(vectors, expected_rows=3, expected_dim=4) is vectors


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4, dtype=np.float32),
        np.ones((2, 4), dtype=np.float64),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.full((2, 4), np.nan, dtype=np.float32),
        np.full((2, 4), np.inf, dtype=np.float32),
        np.full((2, 4), 2.0, dtype=np.float32),
        [[1.0, 0.0]],
    ],
)
def test_invalid_vectors_are_rejected(bad):
    with pytest.raises(EmbeddingError):
        validate_vectors(bad)


def test_shape_expectations_are_enforced():
    with pytest.raises(EmbeddingError):
        validate_vectors(unit(3), expected_rows=2)
    with pytest.raises(EmbeddingError):
        validate_vectors(unit(3), expected_dim=8)


def test_hash_embedder_contract(embedder):
    products = embedder.encode_products(["alpha beta", "gamma"])
    query = embedder.encode_query("alpha")
    validate_vectors(products, expected_rows=2)
    validate_vectors(query, expected_rows=1, expected_dim=products.shape[1])


def test_constructing_real_embedder_does_not_load_a_model():
    embedder = E5Embedder()
    assert embedder._model is None
    assert embedder.model_id == "intfloat/multilingual-e5-small"


@pytest.mark.slow
def test_real_e5_model_roundtrip():
    pytest.importorskip("sentence_transformers")
    real = E5Embedder()
    vectors = real.encode_products(["A light laptop for coding."])
    assert vectors.shape == (1, 384)
    assert real.encode_query("portable laptop").shape == (1, 384)
