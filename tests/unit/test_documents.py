from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.documents import DOCUMENT_FORMAT_VERSION, catalog_documents, product_to_text

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURE)


def test_format_version_is_declared():
    assert DOCUMENT_FORMAT_VERSION == "doc-v1"


def test_text_is_deterministic(catalog):
    assert catalog_documents(catalog) == catalog_documents(catalog)


def test_one_document_per_product_in_row_order(catalog):
    docs = catalog_documents(catalog)
    assert len(docs) == len(catalog)
    for product, doc in zip(catalog.products, docs, strict=True):
        assert doc.startswith(f"Title: {product.title}\n")


def test_each_document_maps_to_exactly_one_title(catalog):
    docs = catalog_documents(catalog)
    for product in catalog.products:
        assert sum(f"Title: {product.title}\n" in doc for doc in docs) == 1


def test_price_and_stock_are_excluded(catalog):
    for product, doc in zip(catalog.products, catalog_documents(catalog), strict=True):
        assert str(product.price_cents) not in doc
        assert "stock" not in doc.lower()
        assert "USD" not in doc


def test_known_values_are_rendered(catalog):
    text = product_to_text(catalog.get("P001"))
    assert "RAM: 16 GB; Weight: 1.30 kg" in text
    assert "GPU: integrated" in text
    assert "Use cases: coding, commuting, office work" in text


def test_unknown_fields_are_explicit(catalog):
    text = product_to_text(catalog.get("P008"))
    assert "Weight: unknown" in text
    assert "0.00" not in text
