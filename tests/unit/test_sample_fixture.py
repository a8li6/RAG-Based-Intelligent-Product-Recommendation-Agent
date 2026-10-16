import json
from pathlib import Path

from recommender.schemas import Product

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"

# (id, price_cents, ram_gb, storage_gb, weight_kg, gpu_type, in_stock) from the reference table
EXPECTED = {
    "P001": (84900, 16, 512, 1.30, "integrated", True),
    "P002": (119900, 32, 1024, 2.20, "dedicated", True),
    "P003": (54900, 8, 256, 1.40, "integrated", True),
    "P004": (69900, 16, 512, 1.20, "integrated", True),
    "P005": (159900, 32, 1024, 2.10, "dedicated", True),
    "P006": (74900, 16, 512, 1.80, "integrated", True),
    "P007": (79900, 16, 512, 1.10, "integrated", False),
    "P008": (64900, 16, 512, None, "integrated", True),
}


def load():
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return [Product.model_validate(json.loads(line)) for line in lines if line.strip()]


def test_fixture_has_eight_unique_valid_products():
    products = load()
    assert len(products) == 8
    assert len({p.product_id for p in products}) == 8


def test_fixture_matches_reference_table():
    for p in load():
        got = (p.price_cents, p.ram_gb, p.storage_gb, p.weight_kg, p.gpu_type, p.in_stock)
        assert got == EXPECTED[p.product_id], p.product_id


def test_fixture_is_synthetic_demolab():
    for p in load():
        assert p.brand == "DemoLab"
        assert ".example/" in p.source_url
