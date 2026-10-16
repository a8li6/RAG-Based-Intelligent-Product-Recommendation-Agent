import json
from pathlib import Path

import pytest

from recommender.catalog import CatalogValidationError, load_catalog
from recommender.cli import main
from recommender.grounding import deterministic_plan, render_items, validate_plan
from recommender.llm import build_messages
from recommender.query import parse_and_merge
from recommender.ranking import build_evidence
from recommender.retrieval import Candidate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_invalid_catalog_reports_every_error_with_its_line():
    with pytest.raises(CatalogValidationError) as err:
        load_catalog(FIXTURES / "invalid_catalog.jsonl")
    by_line = {}
    for issue in err.value.issues:
        by_line.setdefault(issue.line, []).append(issue.message)
    assert set(by_line) == {2, 3, 4, 5, 6, 7}  # line 1 is valid, every other line has an error
    assert "price_cents" in by_line[2][0]
    assert "rating" in by_line[3][0]
    assert "currency" in by_line[4][0]
    assert "duplicate product_id" in by_line[5][0]
    assert "invalid JSON" in by_line[6][0]
    assert "updated_at" in by_line[7][0]


def test_validate_data_cli_lists_all_fixture_errors(capsys):
    assert main(["validate-data", "--input", str(FIXTURES / "invalid_catalog.jsonl")]) == 1
    err = capsys.readouterr().err
    assert "6 error(s)" in err
    for line in range(2, 8):
        assert f"line {line}" in err


def test_hostile_description_is_inert_end_to_end():
    catalog = load_catalog(FIXTURES / "hostile_product.jsonl")
    intent = parse_and_merge("laptop for coding").intent
    selected = [Candidate("P201", 0, 0.9)]
    evidence = build_evidence(selected, intent, catalog)

    user = build_messages(intent, selected, evidence, catalog)[1]["content"]
    assert user.count("</candidates>") == 1
    assert "<system>" not in user

    plan = deterministic_plan(selected, evidence)
    validate_plan(plan, selected, evidence, intent, catalog)
    shown = json.dumps(render_items(plan, evidence, intent, catalog)[0].model_dump())
    assert "IGNORE" not in shown
    assert "P999" not in shown
