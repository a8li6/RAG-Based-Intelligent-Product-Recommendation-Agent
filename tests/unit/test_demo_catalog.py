import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.documents import catalog_documents

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo" / "products.jsonl"
SAMPLE = ROOT / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(DEMO)


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "build_demo_catalog", ROOT / "scripts" / "build_demo_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_catalog_is_valid_and_in_the_documented_size_range(catalog):
    assert len(catalog) == 60
    assert 40 <= len(catalog) <= 100
    assert (
        len({p.title for p in catalog.products}) == 60
    )  # exact-model-name search needs unique titles


def test_committed_file_is_exactly_what_the_script_generates(generator):
    assert DEMO.read_text(encoding="utf-8") == generator.render(generator.build_records())


def test_first_eight_records_are_the_unchanged_reference_fixture():
    fixture = [json.loads(x) for x in SAMPLE.read_text().splitlines() if x.strip()]
    demo = [json.loads(x) for x in DEMO.read_text().splitlines() if x.strip()]
    assert demo[:8] == fixture


def test_catalog_covers_the_segments_the_evaluation_needs(catalog):
    products = catalog.products
    assert sum(p.gpu_type == "dedicated" for p in products) >= 15
    assert sum(not p.in_stock for p in products) >= 5
    assert sum(p.weight_kg is None for p in products) >= 2
    assert sum(p.ram_gb is None for p in products) >= 1
    assert sum(p.gpu_type is None for p in products) >= 1
    assert sum(p.storage_gb is None for p in products) >= 1
    assert min(p.price_cents for p in products) < 50_000 < max(p.price_cents for p in products)
    assert len({p.brand for p in products}) >= 5
    uses = Counter(u for p in products for u in p.use_cases)
    for use in ("coding", "gaming", "study", "travel", "3D work", "office work"):
        assert uses[use] >= 5, use


def test_descriptions_never_contradict_their_own_data(catalog):
    for p in catalog.products:
        text = p.description
        for ram in re.findall(r"(\d+) GB RAM", text):
            assert p.ram_gb == int(ram), p.product_id
        for ssd in re.findall(r"(\d+) GB SSD", text):
            assert p.storage_gb == int(ssd), p.product_id
        for kg in re.findall(r"(\d+\.\d+) kg", text):
            assert p.weight_kg is not None and abs(p.weight_kg - float(kg)) < 0.006, p.product_id
        if p.gpu_type:
            other = "integrated" if p.gpu_type == "dedicated" else "dedicated"
            assert other not in text.lower(), p.product_id
        if p.weight_kg is None:
            assert "weight" in text.lower() and "not provided" in text.lower(), p.product_id


def test_unknown_values_are_stated_as_not_provided(catalog):
    for p in catalog.products:
        for field, label in (
            ("ram_gb", "RAM size"),
            ("storage_gb", "storage size"),
            ("gpu_type", "graphics type"),
        ):
            if getattr(p, field) is None:
                assert label in p.description, (p.product_id, field)


def test_no_benchmark_or_performance_claims(catalog):
    banned = re.compile(
        r"\b(fps|benchmark|score|faster than|hours of battery|battery life|\d+\s*%)", re.IGNORECASE
    )
    for p in catalog.products:
        assert not banned.search(p.description), p.product_id


def test_documents_stay_well_under_the_model_token_limit(catalog):
    assert max(len(doc.split()) for doc in catalog_documents(catalog)) < 120


def test_every_product_id_is_unique_and_ordered(catalog):
    ids = catalog.product_ids
    assert ids == sorted(ids) == [f"P{n:03d}" for n in range(1, 61)]


def test_the_data_card_facts_match_the_catalog(catalog):
    card = (ROOT / "docs" / "data_card.md").read_text(encoding="utf-8")
    assert f"`{catalog.sha256[:12]}" in card
    assert "| Records | 60" in card
    brands = Counter(p.brand for p in catalog.products)
    for brand, n in brands.items():
        assert f"{brand} ({n})" in card, brand
    assert f"{sum(p.in_stock for p in catalog.products)} in stock" in card
    assert f"{sum(not p.in_stock for p in catalog.products)} out of stock" in card
    assert f"{sum(p.gpu_type == 'integrated' for p in catalog.products)} integrated" in card
    assert f"{sum(p.gpu_type == 'dedicated' for p in catalog.products)} dedicated" in card
    for field, label in (
        ("ram_gb", "ram_gb"),
        ("storage_gb", "storage_gb"),
        ("weight_kg", "weight_kg"),
        ("gpu_type", "gpu_type"),
    ):
        missing = [p.product_id for p in catalog.products if getattr(p, field) is None]
        row = next(
            line
            for line in card.splitlines()
            if line.startswith(f"| `{label}` | ") and "product" in line
        )
        for pid in missing:
            assert pid in row, (field, pid)
