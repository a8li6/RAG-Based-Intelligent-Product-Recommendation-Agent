"""Exact semantic retrieval restricted to the eligible subset.

Filter first, then search the complete eligible subset, so the best semantic matches
among eligible products are never lost to a global top-k cut followed by filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from recommender.embeddings import Embedder
from recommender.index import Artifacts, ensure_embedder_matches
from recommender.schemas import Constraints


@dataclass(frozen=True)
class Candidate:
    product_id: str
    row: int
    score: float


def search_eligible(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    eligible_rows: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """Return up to k (row, inner-product score) pairs, best first, among eligible rows.

    Vectors are unit-normalized, so the inner product equals cosine similarity.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if not eligible_rows:
        return []
    rows = np.unique(np.asarray(eligible_rows, dtype=np.int64))
    if rows[0] < 0 or rows[-1] >= len(vectors):
        raise ValueError("eligible row out of range")
    query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    if query.shape[1] != vectors.shape[1]:
        raise ValueError("query dimension does not match the vectors")

    subset = np.ascontiguousarray(vectors[rows], dtype=np.float32)
    local = faiss.IndexFlatIP(subset.shape[1])
    local.add(subset)
    scores, positions = local.search(query, min(k, len(rows)))
    return [
        (int(rows[pos]), float(score))
        for score, pos in zip(scores[0], positions[0], strict=True)
        if 0 <= pos < len(rows)
    ]


class Retriever:
    """Embeds a query once and searches the eligible subset of a loaded artifact set."""

    def __init__(self, artifacts: Artifacts, embedder: Embedder, retrieval_k: int = 10):
        ensure_embedder_matches(artifacts, embedder)
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be positive")
        self.artifacts = artifacts
        self.embedder = embedder
        self.retrieval_k = retrieval_k

    def eligible_rows(self, constraints: Constraints) -> list[int]:
        return self.artifacts.catalog.eligible_rows(constraints)

    def search(self, query: str, eligible_rows: list[int]) -> list[Candidate]:
        if not eligible_rows:
            return []
        query_vector = self.embedder.encode_query(query)
        hits = search_eligible(
            query_vector, self.artifacts.vectors, eligible_rows, self.retrieval_k
        )
        products = self.artifacts.catalog.products
        return [Candidate(products[row].product_id, row, score) for row, score in hits]
