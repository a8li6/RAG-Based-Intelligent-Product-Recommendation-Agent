import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FA = ROOT / "reports" / "fa"
NAMES = [
    "baseline-no-persian-support-dev.json",
    "baseline-no-persian-support-test.json",
    "eval-dev.json",
    "eval-test.json",
]


@pytest.fixture(scope="module")
def reports():
    return {n: json.loads((FA / n).read_text(encoding="utf-8")) for n in NAMES}


def test_every_persian_report_matches_the_committed_persian_set(reports):
    queries_sha = hashlib.sha256(
        (ROOT / "data" / "eval" / "fa" / "queries.jsonl").read_bytes()
    ).hexdigest()
    meta = json.loads((ROOT / "data" / "eval" / "fa" / "meta.json").read_text(encoding="utf-8"))
    for name, rep in reports.items():
        assert rep["configuration"]["eval_set_sha256"] == queries_sha, name
        assert rep["configuration"]["test_labels_sha256"] == meta["test_labels_sha256"], name
        assert rep["configuration"]["catalog_sha256"] == meta["catalog_sha256"], name


def test_the_baseline_shows_the_defect_and_the_final_run_shows_it_fixed(reports):
    for split in ("dev", "test"):
        before = reports[f"baseline-no-persian-support-{split}.json"]["summary"]["system"]
        after = reports[f"eval-{split}.json"]["summary"]["system"]
        assert before["constraint_violation_rate"]["numerator"] > 0
        assert after["constraint_violation_rate"]["numerator"] == 0
        assert after["interpretation_accuracy"]["value"] == 1.0
        assert before["interpretation_accuracy"]["value"] < 1.0


def test_the_persian_reports_use_the_same_default_configuration_as_english(reports):
    en = json.loads((ROOT / "reports" / "eval-test-no-cutoff.json").read_text(encoding="utf-8"))
    for name in ("eval-dev.json", "eval-test.json"):
        cfg = reports[name]["configuration"]
        assert (
            cfg["settings"]["min_similarity"] is None
            and cfg["settings"]["explanation_mode"] == "deterministic"
        )
        assert (
            cfg["embedder_revision"] == en["configuration"]["embedder_revision"]
        )  # same fitted state


def test_no_persian_report_shows_a_constraint_violation_after_support(reports):
    for name in ("eval-dev.json", "eval-test.json"):
        assert all(q["violations"] == [] for q in reports[name]["per_query"])


def test_the_latency_pair_report_is_consistent():
    lat = json.loads((FA / "latency-pairs.json").read_text(encoding="utf-8"))
    assert lat["pairs"] == len([p for p in lat["per_query"] if p["expected_status"] == "ok"])
    assert lat["ratio_q1"] <= lat["ratio_median"] <= lat["ratio_q3"]


@pytest.mark.parametrize("name", ["evaluation_fa.md", "fa_review_sheet.md"])
def test_generated_persian_documents_are_exactly_what_the_script_produces(name):
    paths = [ROOT / "docs" / "evaluation_fa.md", ROOT / "docs" / "fa_review_sheet.md"]
    before = {p: p.read_text(encoding="utf-8") for p in paths}
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "render_evaluation_fa_md.py")],
            check=True,
            capture_output=True,
            cwd=ROOT,
        )
        assert (ROOT / "docs" / name).read_text(encoding="utf-8") == before[ROOT / "docs" / name], (
            f"{name} is out of date"
        )
    finally:
        for p, text in before.items():
            p.write_text(text, encoding="utf-8")


def test_the_document_and_sheet_state_the_review_status_up_front():
    import re

    raw = (ROOT / "docs" / "evaluation_fa.md").read_text(encoding="utf-8")[:1800]
    doc = re.sub(r"[>*\s]+", " ", raw).lower()  # ignore blockquote markers, bold and line breaks
    assert "written by an ai assistant" in doc and "not been reviewed" in doc and "native" in doc
    assert "english" in doc  # responses stay English
    sheet = (ROOT / "docs" / "fa_review_sheet.md").read_text(encoding="utf-8")
    assert "native Persian speaker" in sheet
    assert (
        len(
            [
                line
                for line in sheet.splitlines()
                if line.startswith("| ") and "---" not in line and "Persian query" not in line
            ]
        )
        == 60
    )


def test_the_readme_persian_numbers_match_the_reports(reports):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    after, before = (
        reports["eval-test.json"]["summary"],
        reports["baseline-no-persian-support-test.json"]["summary"],
    )
    for rep in (after, before):
        for key in ("constraint_violation_rate", "interpretation_accuracy"):
            v = rep["system"][key]
            assert f"{v['numerator']}/{v['denominator']}" in readme, key
        assert f"{rep['retrieval']['builtin']['recall_at_10']['mean']:.3f}" in readme
    assert "not been" in readme and "native speaker" in readme
    assert (
        "docs/images/ui-persian.png" in readme
        and (ROOT / "docs" / "images" / "ui-persian.png").exists()
    )
