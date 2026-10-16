import math

import pytest
from pydantic import ValidationError

from recommender.schemas import Product


def make_record(**overrides):
    record = {
        "product_id": "P001",
        "title": "Atlas Air 14",
        "category": "laptop",
        "brand": "DemoLab",
        "price_cents": 84900,
        "currency": "USD",
        "in_stock": True,
        "ram_gb": 16,
        "storage_gb": 512,
        "weight_kg": 1.3,
        "gpu_type": "integrated",
        "use_cases": ["coding", "commuting"],
        "description": "A light laptop.",
        "source_url": "https://catalog.example/products/P001",
        "updated_at": "2026-10-01T00:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_valid_record_parses():
    product = Product.model_validate(make_record())
    assert product.price_cents == 84900
    assert product.updated_at.utcoffset() is not None


def test_unknown_specs_stay_none():
    product = Product.model_validate(make_record(weight_kg=None, ram_gb=None, gpu_type=None))
    assert product.weight_kg is None
    assert product.ram_gb is None
    assert product.gpu_type is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_cents": -1},
        {"price_cents": 849.5},
        {"price_cents": "84900"},
        {"currency": "EUR"},
        {"category": "phone"},
        {"product_id": "X001"},
        {"product_id": "P1"},
        {"title": ""},
        {"title": "x" * 121},
        {"in_stock": "yes"},
        {"ram_gb": 0},
        {"weight_kg": 0},
        {"weight_kg": float("nan")},
        {"weight_kg": math.inf},
        {"weight_kg": 50.0},
        {"gpu_type": "quantum"},
        {"use_cases": []},
        {"use_cases": ["  "]},
        {"source_url": "http://catalog.example/p"},
        {"source_url": "not a url"},
        {"updated_at": "2026-10-01T00:00:00"},
        {"updated_at": "yesterday"},
    ],
)
def test_malformed_records_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Product.model_validate(make_record(**overrides))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Product.model_validate(make_record(rating=4.8))


def test_text_is_stripped_and_unicode_normalized():
    product = Product.model_validate(make_record(title="  Café Book  "))
    assert product.title == "Café Book"
