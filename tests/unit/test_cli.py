import json
import subprocess
import sys
from pathlib import Path

from recommender.cli import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "sample" / "products.jsonl"


def test_validate_data_ok(capsys):
    assert main(["validate-data", "--input", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "8 records" in out
    assert "sha256=" in out


def test_validate_data_invalid_file_exits_nonzero(tmp_path, capsys):
    record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    record["price_cents"] = -1
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert main(["validate-data", "--input", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "line 1" in err
    assert "P001" in err


def test_validate_data_missing_file(tmp_path, capsys):
    assert main(["validate-data", "--input", str(tmp_path / "nope.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_module_entry_point_exit_code(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "recommender.cli", "validate-data", "--input", str(bad)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "invalid JSON" in result.stderr


def test_build_index_and_search_roundtrip(tmp_path, capsys, embedder):
    out = tmp_path / "catalog-v1"

    def factory(settings):
        return embedder

    assert main(["build-index", "--input", str(FIXTURE), "--output", str(out)], factory) == 0
    assert "built 8 products" in capsys.readouterr().out

    args = ["search", "--query", "portable laptop for coding", "--artifacts", str(out)]
    args += ["--max-price-cents", "90000", "--min-ram-gb", "16", "--max-weight-kg", "1.5"]
    assert main(args, factory) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "eligible: 2 of 8 products"
    assert {line.split()[0] for line in lines[1:]} == {"P001", "P004"}


def test_search_without_artifacts_exits_nonzero(tmp_path, capsys, embedder):
    args = ["search", "--query", "x", "--artifacts", str(tmp_path / "missing")]
    assert main(args, lambda settings: embedder) == 1
    assert "rebuild" in capsys.readouterr().err


def test_build_index_rejects_invalid_catalog(tmp_path, capsys, embedder):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    code = main(
        ["build-index", "--input", str(bad), "--output", str(tmp_path / "o")],
        lambda settings: embedder,
    )
    assert code == 1
    assert not (tmp_path / "o").exists()


def test_build_index_failure_exits_nonzero(tmp_path, capsys):
    def broken(settings):
        raise RuntimeError("model unavailable")

    code = main(["build-index", "--input", str(FIXTURE), "--output", str(tmp_path / "o")], broken)
    assert code == 1
    assert "model unavailable" in capsys.readouterr().err
