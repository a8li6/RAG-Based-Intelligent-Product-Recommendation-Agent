"""Deterministic search documents: one concise document per product.

Price and stock are deliberately excluded. They are volatile, enforced by exact code,
and must not force re-embedding when they change.
"""

from __future__ import annotations

from recommender.catalog import Catalog
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
