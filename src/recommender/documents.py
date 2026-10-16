"""Deterministic search documents: one concise document per product.

Price and stock are deliberately excluded. They are volatile, enforced by exact code,
and must not force re-embedding when they change.
"""

from __future__ import annotations

from collections.abc import Callable

from recommender.catalog import Catalog
from recommender.embeddings import PASSAGE_PREFIX
from recommender.schemas import Product

# Bump whenever the text produced by product_to_text changes: embeddings become invalid.
DOCUMENT_FORMAT_VERSION = "doc-v1"


def product_to_text(product: Product) -> str:
    ram = "unknown" if product.ram_gb is None else f"{product.ram_gb} GB"
    weight = "unknown" if product.weight_kg is None else f"{product.weight_kg:.2f} kg"
    return "\n".join(
        [
            f"Title: {product.title}",
            f"Category: {product.category}; Brand: {product.brand}",
            f"RAM: {ram}; Weight: {weight}",
            f"GPU: {product.gpu_type or 'unknown'}",
            f"Use cases: {', '.join(product.use_cases)}",
            f"Description: {product.description}",
        ]
    )


def catalog_documents(catalog: Catalog) -> list[str]:
    """One document per product, in the catalog's stable ID-sorted row order."""
    return [product_to_text(product) for product in catalog.products]


DEFAULT_MAX_TOKENS = 512  # multilingual-e5-small input limit
TARGET_TOKEN_RANGE = (250, 350)  # starting target from the design; measured, not enforced


class DocumentTooLongError(ValueError):
    """Documents exceed the embedding model's input limit. Nothing is silently truncated."""

    def __init__(self, too_long: dict[str, int], limit: int):
        self.too_long = too_long
        self.limit = limit
        listing = ", ".join(f"{pid} ({n} tokens)" for pid, n in sorted(too_long.items()))
        super().__init__(
            f"{len(too_long)} document(s) exceed the {limit}-token limit: {listing}. "
            "Shorten the description; truncation would drop identity or specifications."
        )


def check_document_lengths(
    catalog: Catalog,
    count_tokens: Callable[[str], int],
    limit: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Count tokens as embedded (passage prefix included) and reject any over the limit.

    Returns {"count", "max", "mean", "limit"} for the manifest and the build report.
    """
    counts = {
        product.product_id: count_tokens(PASSAGE_PREFIX + text)
        for product, text in zip(catalog.products, catalog_documents(catalog), strict=True)
    }
    too_long = {pid: n for pid, n in counts.items() if n > limit}
    if too_long:
        raise DocumentTooLongError(too_long, limit)
    values = list(counts.values())
    return {
        "count": len(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "limit": limit,
    }
