"""Builds the configured embedder. The default (builtin) needs no download or service."""

from __future__ import annotations

from collections.abc import Callable

from recommender.config import Settings
from recommender.embeddings import E5Embedder, Embedder
from recommender.local_embedder import LocalEmbedder

EmbedderFactory = Callable[[Settings], Embedder]


def default_embedder_factory(settings: Settings) -> Embedder:
    if settings.embedding_backend == "builtin":
        return LocalEmbedder()
    return E5Embedder(
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
