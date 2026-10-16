"""Shared test doubles. Unit tests never download models or call a live service."""

import hashlib
import re

import numpy as np
import pytest

DIM = 64


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    model_id = "test/hash-embedder"
    revision = "test-rev-1"

    max_tokens = 512

    def count_tokens(self, text):
        return len(re.findall(r"\S+", text))

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float64)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % DIM] += 1.0
        if not vec.any():
            vec[0] = 1.0
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def encode_products(self, texts):
        return np.ascontiguousarray([self._vec("passage: " + t) for t in texts], dtype=np.float32)

    def encode_query(self, text):
        return np.ascontiguousarray([self._vec("query: " + text)], dtype=np.float32)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def settings(monkeypatch):
    from recommender.config import Settings, load_settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    return load_settings()
