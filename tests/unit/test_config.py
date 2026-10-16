from pathlib import Path

import pytest

from recommender.config import PROJECT_ROOT, ConfigError, Settings, load_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in list(Settings.model_fields):
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def test_defaults_are_valid():
    s = load_settings()
    assert s.retrieval_k == 10
    assert s.max_recommendations == 3
    assert s.default_currency == "USD"
    assert s.explanation_mode == "llm"
    assert s.llm_base_url == "http://127.0.0.1:11434"


def test_paths_resolve_from_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    assert s.catalog_path == PROJECT_ROOT / "data/sample/products.jsonl"
    assert s.catalog_path.is_file()
    assert s.artifact_dir.is_absolute()


def test_absolute_paths_are_kept(tmp_path):
    assert load_settings(catalog_path=tmp_path / "c.jsonl").catalog_path == tmp_path / "c.jsonl"


def test_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "5")
    monkeypatch.setenv("EXPLANATION_MODE", "deterministic")
    s = load_settings()
    assert (s.retrieval_k, s.explanation_mode) == (5, "deterministic")


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": -1},
        {"default_currency": "EUR"},
        {"explanation_mode": "freeform"},
        {"retrieval_k": 0},
        {"retrieval_k": 1000},
        {"max_recommendations": 4},
        {"embedding_batch_size": 0},
        {"llm_base_url": "ftp://host"},
        {"log_level": "CHATTY"},
    ],
)
def test_invalid_settings_fail_with_clear_error(overrides):
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_settings(**overrides)


def test_error_message_names_the_field():
    with pytest.raises(ConfigError, match="llm_timeout_seconds"):
        load_settings(llm_timeout_seconds=0)


def test_env_example_matches_defaults():
    lines = (PROJECT_ROOT / ".env.example").read_text().splitlines()
    example = dict(line.split("=", 1) for line in lines if "=" in line)
    s = Settings.model_validate({k.lower(): v for k, v in example.items()})
    defaults = load_settings()
    for name in Settings.model_fields:
        assert getattr(s, name) == getattr(defaults, name), name
    assert set(example) <= {n.upper() for n in Settings.model_fields}
    assert isinstance(s.catalog_path, Path)
