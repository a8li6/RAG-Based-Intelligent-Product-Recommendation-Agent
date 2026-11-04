import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
CHECK = ROOT / "scripts" / "check.sh"


def test_actions_are_pinned_to_full_commit_shas():
    uses = re.findall(r"uses:\s*(\S+)", WORKFLOW)
    assert uses, "the workflow should use actions"
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), f"not SHA-pinned: {ref}"


def test_workflow_is_read_only_and_uses_no_secrets_or_risky_triggers():
    assert re.search(r"^permissions:\n\s+contents: read\s*$", WORKFLOW, re.MULTILINE)
    assert "${{ secrets" not in WORKFLOW  # no secret is ever referenced
    assert "pull_request_target" not in WORKFLOW
    assert "persist-credentials: false" in WORKFLOW


def test_workflow_installs_the_lock_then_the_project_without_dependencies():
    lock = WORKFLOW.index("pip install -r requirements-lock.txt")
    editable = WORKFLOW.index("pip install --no-deps -e .")
    assert lock < editable
    assert 'python-version: "3.11"' in WORKFLOW
    assert "cache-dependency-path: requirements-lock.txt" in WORKFLOW


def test_workflow_runs_on_pull_requests_and_pushes_to_main():
    assert re.search(r"push:\n\s+branches: \[main\]", WORKFLOW)
    assert re.search(r"^\s+pull_request:", WORKFLOW, re.MULTILINE)


def test_workflow_delegates_to_the_check_script():
    assert "bash scripts/check.sh" in WORKFLOW


def test_check_script_is_strict_and_covers_every_gate():
    text = CHECK.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    for gate in (
        "pip check", "ruff check", "ruff format --check", "validate-data --input data/sample",
        "validate-data --input data/demo", "pytest -m \"not slow\"", "evaluate --split dev",
    ):  # fmt: skip
        assert gate in text, gate
    assert CHECK.stat().st_mode & stat.S_IXUSR


def test_the_invalid_catalog_gate_would_really_fail_the_job():
    text = CHECK.read_text(encoding="utf-8")
    assert "tests/fixtures/invalid_catalog.jsonl" in text
    assert "exit 1" in text  # the script exits nonzero if the invalid catalog is accepted
