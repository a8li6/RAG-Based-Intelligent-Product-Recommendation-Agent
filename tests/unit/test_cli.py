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
