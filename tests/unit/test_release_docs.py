import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
NOTES = (ROOT / "docs" / "portfolio_notes.md").read_text(encoding="utf-8")
ACCEPT = (ROOT / "docs" / "acceptance_checklist.md").read_text(encoding="utf-8")
RELEASE = (ROOT / "docs" / "release_notes_v0.1.0.md").read_text(encoding="utf-8")


def test_portfolio_numbers_match_the_reports():
    report = json.loads((ROOT / "reports" / "eval-test-no-cutoff.json").read_text())
    r, s = report["summary"]["retrieval"], report["summary"]["system"]
    for method in ("builtin", "post_filter", "keyword"):
        assert f"{r[method]['recall_at_10']['mean']:.2f}" in NOTES, method
    for key in ("constraint_violation_rate", "evidence_validity"):
        assert f"{s[key]['numerator']}/{s[key]['denominator']}" in NOTES, key
        assert f"{s[key]['numerator']}/{s[key]['denominator']}" in ACCEPT, key


def test_portfolio_notes_warn_against_overclaiming():
    lowered = NOTES.lower()
    assert "do not claim" in lowered
    assert "neural" in lowered and "language model" in lowered and "synthetic" in lowered
    assert "single-author" in lowered or "one author" in lowered


def test_the_checklist_does_not_tick_what_was_not_done():
    assert "**not done**" in ACCEPT  # the real-model checks
    assert "never run" in ACCEPT.lower()
    assert "needs your confirmation" in ACCEPT  # repository visibility is the owner's call


def test_release_notes_separate_verified_from_unverified():
    assert "## Verified" in RELEASE and "## Not verified / known limits" in RELEASE
    assert "never run against" in RELEASE


def test_the_tag_is_not_claimed_as_done_while_it_is_not_on_the_remote():
    row = next(
        line for line in ACCEPT.splitlines() if "release tag and short release notes" in line
    )
    assert "| partial |" in row and "not created" in row
    assert "Publishing the tag" in ACCEPT and "git push origin v0.1.0" in ACCEPT
    progress = (ROOT / "docs" / "progress.md").read_text(encoding="utf-8")
    assert "was **not** pushed" in progress
