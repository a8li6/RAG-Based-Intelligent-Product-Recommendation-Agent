import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from recommender.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NAMES = ["eval-dev.json", "eval-test.json", "eval-test-no-cutoff.json"]


@pytest.fixture(scope="module")
def reports():
    return {n: json.loads((REPORTS / n).read_text(encoding="utf-8")) for n in NAMES}


def test_every_report_is_current_for_the_committed_data(reports):
    catalog_sha = load_catalog(ROOT / "data" / "demo" / "products.jsonl").sha256
    queries_sha = hashlib.sha256(
        (ROOT / "data" / "eval" / "queries.jsonl").read_bytes()
    ).hexdigest()
    meta = json.loads((ROOT / "data" / "eval" / "meta.json").read_text())
    for name, report in reports.items():
        cfg = report["configuration"]
        assert cfg["catalog_sha256"] == catalog_sha, name
        assert cfg["eval_set_sha256"] == queries_sha, name
        assert cfg["test_labels_sha256"] == meta["test_labels_sha256"], name


def test_reports_cover_the_right_splits(reports):
    assert reports["eval-dev.json"]["split"] == "dev"
    assert len(reports["eval-dev.json"]["per_query"]) == 20
    for name in ("eval-test.json", "eval-test-no-cutoff.json"):
        assert reports[name]["split"] == "test"
        assert len(reports[name]["per_query"]) == 40


def test_the_cutoff_configuration_is_recorded_in_each_report(reports):
    assert reports["eval-dev.json"]["configuration"]["settings"]["min_similarity"] is None
    assert (
        reports["eval-test-no-cutoff.json"]["configuration"]["settings"]["min_similarity"] is None
    )
    calibrated = reports["eval-test.json"]["configuration"]["settings"]["min_similarity"]
    assert calibrated == pytest.approx(0.0603, abs=1e-4)
    assert calibrated == pytest.approx(
        reports["eval-dev.json"]["calibration"]["threshold"], abs=1e-4
    )


def test_no_report_ever_shows_a_constraint_violation(reports):
    for name, report in reports.items():
        assert all(q["violations"] == [] for q in report["per_query"]), name


def test_the_default_shipped_configuration_keeps_the_cutoff_off():
    from recommender.config import Settings

    assert Settings.model_fields["min_similarity"].default is None


def test_the_evaluation_document_is_exactly_what_the_reports_generate():
    doc = ROOT / "docs" / "evaluation.md"
    before = doc.read_text(encoding="utf-8")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "render_evaluation_md.py")],
            check=True, capture_output=True, cwd=ROOT,
        )  # fmt: skip
        assert doc.read_text(encoding="utf-8") == before, "docs/evaluation.md is out of date"
    finally:
        doc.write_text(before, encoding="utf-8")


def test_the_document_states_its_limits_up_front():
    text = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    head = text[:1800]
    for phrase in ("synthetic", "one author", "optimistic", "no confidence"):
        assert phrase in head.lower() or phrase.replace(" ", "\n") in head.lower(), phrase
