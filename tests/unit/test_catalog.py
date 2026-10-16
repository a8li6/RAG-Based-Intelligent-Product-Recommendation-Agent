import json
from pathlib import Path

import pytest

from recommender.catalog import CatalogValidationError, load_catalog, parse_catalog
from recommender.schemas import Constraints

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURE)


def eligible_ids(catalog, **kwargs):
    rows = catalog.eligible_rows(Constraints(**kwargs))
    return {catalog.products[row].product_id for row in rows}


def test_reference_case_budget_ram_weight_stock(catalog):
    ids = eligible_ids(catalog, max_price_cents=90000, min_ram_gb=16, max_weight_kg=1.5)
    assert ids == {"P001", "P004"}


def test_reference_case_budget_ram_no_weight_limit(catalog):
    assert eligible_ids(catalog, max_price_cents=70000, min_ram_gb=16) == {"P004", "P008"}


def test_adding_weight_limit_excludes_unknown_weight(catalog):
    ids = eligible_ids(catalog, max_price_cents=70000, min_ram_gb=16, max_weight_kg=1.5)
    assert ids == {"P004"}


def test_no_match_case(catalog):
    assert eligible_ids(catalog, max_price_cents=30000, min_ram_gb=16) == set()


def test_price_boundary_is_inclusive(catalog):
    assert "P004" in eligible_ids(catalog, max_price_cents=69900)
    assert "P004" not in eligible_ids(catalog, max_price_cents=69899)


def test_ram_boundary_is_inclusive(catalog):
    assert "P003" not in eligible_ids(catalog, min_ram_gb=16)
    assert "P003" in eligible_ids(catalog, min_ram_gb=8)


def test_weight_boundary_is_inclusive(catalog):
    assert "P004" in eligible_ids(catalog, max_weight_kg=1.2)
    assert "P004" not in eligible_ids(catalog, max_weight_kg=1.19)


def test_stock_requirement_toggle(catalog):
    assert "P007" not in eligible_ids(catalog)
    assert "P007" in eligible_ids(catalog, require_in_stock=False)


def test_unconstrained_stock_only_returns_seven(catalog):
    assert len(eligible_ids(catalog)) == 7


def test_rows_are_id_sorted_positions(catalog):
    assert catalog.product_ids == sorted(catalog.product_ids)
    rows = catalog.eligible_rows(
        Constraints(max_price_cents=90000, min_ram_gb=16, max_weight_kg=1.5)
    )
    assert rows == sorted(rows)


def test_hash_is_deterministic_and_order_independent(catalog):
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    shuffled = parse_catalog("\n".join(reversed(lines)))
    assert shuffled.sha256 == catalog.sha256
    assert len(catalog.sha256) == 64


def test_hash_changes_when_content_changes(catalog):
    lines = [json.loads(x) for x in FIXTURE.read_text(encoding="utf-8").splitlines()]
    lines[0]["price_cents"] += 1
    changed = parse_catalog("\n".join(json.dumps(x) for x in lines))
    assert changed.sha256 != catalog.sha256


def test_duplicate_ids_are_rejected_with_line_numbers(catalog):
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    with pytest.raises(CatalogValidationError) as err:
        parse_catalog("\n".join([*lines, lines[0]]))
    assert "duplicate product_id" in str(err.value)
    assert err.value.issues[0].line == 9
    assert err.value.issues[0].product_id == "P001"


def test_malformed_json_reports_line(catalog):
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    with pytest.raises(CatalogValidationError) as err:
        parse_catalog("\n".join([lines[0], "{not json"]))
    assert err.value.issues[0].line == 2


def test_all_errors_are_collected_not_just_first():
    bad1 = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    bad2 = dict(bad1, product_id="P002")
    bad1["price_cents"] = -5
    bad2["currency"] = "EUR"
    with pytest.raises(CatalogValidationError) as err:
        parse_catalog(json.dumps(bad1) + "\n" + json.dumps(bad2))
    assert {issue.line for issue in err.value.issues} == {1, 2}


def test_empty_catalog_is_rejected():
    with pytest.raises(CatalogValidationError):
        parse_catalog("\n\n")


def test_report_summarizes_the_catalog(catalog):
    from recommender.catalog import catalog_report

    report = catalog_report(catalog, FIXTURE)
    assert report["record_count"] == 8
    assert report["catalog_sha256"] == catalog.sha256
    assert report["missing_fields"] == {
        "ram_gb": 0, "storage_gb": 0, "weight_kg": 1, "gpu_type": 0,
    }  # fmt: skip
    assert report["in_stock_count"] == 7
    assert report["categories"] == ["laptop"]
    assert report["currencies"] == ["USD"]
    assert report["source"]["file"] == "products.jsonl"
    assert report["source"]["bytes"] == FIXTURE.stat().st_size
    assert len(report["source"]["sha256"]) == 64
    assert "records sorted by product_id" in report["transformations"]


def test_report_without_a_source_has_no_source_section(catalog):
    from recommender.catalog import catalog_report

    assert "source" not in catalog_report(catalog)
