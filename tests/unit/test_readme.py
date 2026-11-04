import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_relative_links_and_images_point_to_real_files():
    targets = re.findall(r"\]\((?!https?://|#)([^)\s]+)\)", README)
    assert targets
    for target in targets:
        assert (ROOT / target.split("#")[0]).exists(), target


def test_every_path_in_the_layout_and_quick_start_exists():
    for path in (
        "requirements-lock.txt", "data/demo/products.jsonl", "data/sample/products.jsonl",
        "data/eval/queries.jsonl", "ui/app.py", "scripts/check.sh", ".env.example", "reports",
        "docs/architecture.md", "docs/evaluation.md", "docs/data_card.md",
    ):  # fmt: skip
        assert (ROOT / path).exists(), path
        assert path in README or path.split("/")[0] in README, path


def test_numbers_in_the_readme_match_the_reports():
    report = json.loads((ROOT / "reports" / "eval-test-no-cutoff.json").read_text())
    s, op = report["summary"], report["operational"]
    r, sysm = s["retrieval"], s["system"]
    for method in ("builtin", "keyword", "post_filter", "random"):
        for metric in ("recall_at_10", "mrr_at_3", "ndcg_at_3"):
            assert f"{r[method][metric]['mean']:.3f}" in README, (method, metric)
    assert f"{r['oracle_recall_at_10']['mean']:.3f}" in README
    for key in (
        "constraint_violation_rate",
        "evidence_validity",
        "interpretation_accuracy",
        "status_accuracy",
        "abstention_on_irrelevant",
    ):
        v = sysm[key]
        assert f"{v['numerator']}/{v['denominator']}" in README, key
    assert f"p50 {op['warm_p50_ms']:.2f} ms, p95 {op['warm_p95_ms']:.2f} ms" in README


def test_readme_states_its_scope_and_limits_near_the_top():
    head = README[:1800].lower()
    assert "synthetic" in head
    assert "no neural model and no language model" in head
    assert "not verified against real models" in head


def test_readme_does_not_overclaim():
    lowered = README.lower()
    for phrase in (
        "state-of-the-art",
        "production-ready",
        "hallucination-free",
        "guarantees accuracy",
        "real shoppers",
    ):
        if phrase == "real shoppers":
            assert "not evidence of real-world accuracy" in lowered
        else:
            assert phrase not in lowered, phrase


def test_the_documented_defaults_match_the_code():
    from recommender.config import Settings

    defaults = {n: f.default for n, f in Settings.model_fields.items()}
    assert str(defaults["catalog_path"]) == "data/demo/products.jsonl"
    assert str(defaults["artifact_dir"]) == "artifacts/demo-v1"
    assert defaults["embedding_backend"] == "builtin"
    assert defaults["explanation_mode"] == "deterministic"
    assert defaults["min_similarity"] is None
    for needle in (
        "`data/demo/products.jsonl`",
        "`artifacts/demo-v1`",
        "`builtin`",
        "`deterministic`",
    ):
        assert needle in README, needle


def test_the_ci_badge_points_at_the_real_workflow_file():
    assert "actions/workflows/ci.yml/badge.svg" in README
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists()
