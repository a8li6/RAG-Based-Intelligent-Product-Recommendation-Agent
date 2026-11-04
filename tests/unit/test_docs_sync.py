from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")


def test_every_module_is_described_in_the_architecture_document():
    for path in sorted((ROOT / "src" / "recommender").glob("*.py")):
        if path.name == "__init__.py":
            continue
        assert f"`{path.name}`" in ARCH, f"{path.name} is not in docs/architecture.md"


def test_the_architecture_document_has_no_stale_module_names():
    import re

    named = set(re.findall(r"`([a-z_]+\.py)`", ARCH))
    real = {p.name for p in (ROOT / "src" / "recommender").glob("*.py")}
    assert named - real - {"app.py"} == set(), named - real


def test_the_architecture_document_states_the_scope_honestly():
    head = ARCH[:1500].lower()
    assert "no neural model and no" in head and "language model" in head
    assert "not verified against real models" in head


def test_the_recall_figures_in_the_architecture_match_the_reports():
    import json

    report = json.loads((ROOT / "reports" / "eval-test.json").read_text())
    r = report["summary"]["retrieval"]
    assert f"{r['builtin']['recall_at_10']['mean']:.3f}" in ARCH
    assert f"{r['post_filter']['recall_at_10']['mean']:.3f}" in ARCH
