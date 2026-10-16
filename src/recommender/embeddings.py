"""Embedding adapter. One model, one prefix convention, normalized float32 vectors.

Importing this module never downloads or loads a model; loading happens on first use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# E5 model-card convention: the same prefixes must be used at build time and query time.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
NORM_TOLERANCE = 1e-3


class EmbeddingError(ValueError):
    """Raised when vectors violate the embedding contract."""


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode_products(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def validate_vectors(
    vectors: np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    tolerance: float = NORM_TOLERANCE,
) -> np.ndarray:
    """Require a finite, non-zero, unit-norm float32 matrix of shape (rows, dim)."""
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise EmbeddingError("vectors must be a 2D numpy array of shape (rows, dim)")
    if vectors.dtype != np.float32:
        raise EmbeddingError(f"vectors must be float32, got {vectors.dtype}")
    rows, dim = vectors.shape
    if rows == 0 or dim == 0:
        raise EmbeddingError("vectors must not be empty")
    if expected_rows is not None and rows != expected_rows:
        raise EmbeddingError(f"expected {expected_rows} rows, got {rows}")
    if expected_dim is not None and dim != expected_dim:
        raise EmbeddingError(f"expected dimension {expected_dim}, got {dim}")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("vectors contain NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise EmbeddingError("vectors contain a zero-norm row")
    if not np.allclose(norms, 1.0, atol=tolerance):
        raise EmbeddingError(f"vectors are not unit-normalized (tolerance {tolerance})")
    return vectors


class E5Embedder:
    """multilingual-e5 style encoder via sentence-transformers, loaded once per process."""

    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-small",
        revision: str = "main",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, revision=self.revision, device=self.device
            )
        return self._model

    def _encode(self, prefixed: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return validate_vectors(np.ascontiguousarray(vectors, dtype=np.float32))

    def encode_products(self, texts: list[str]) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + text for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])
