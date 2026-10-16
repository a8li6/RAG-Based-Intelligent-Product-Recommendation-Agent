"""Validated application settings. Configuration is a contract: bad values fail at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Raised with a readable message when settings are invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore", case_sensitive=False
    )

    catalog_path: Path = Path("data/sample/products.jsonl")
    artifact_dir: Path = Path("artifacts/catalog-v1")
    embedding_model: str = Field(default="intfloat/multilingual-e5-small", min_length=1)
    embedding_revision: str = Field(default="main", min_length=1)
    embedding_device: str = Field(default="cpu", min_length=1)
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = Field(default="qwen3:4b", min_length=1)
    llm_timeout_seconds: float = Field(default=60, gt=0, le=600)
    retrieval_k: int = Field(default=10, ge=1, le=100)
    max_recommendations: int = Field(default=3, ge=1, le=3)
    default_currency: Literal["USD"] = "USD"
    explanation_mode: Literal["llm", "deterministic"] = "llm"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_query_chars: int = Field(default=2000, ge=1, le=10_000)

    @field_validator("llm_base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("catalog_path", "artifact_dir")
    @classmethod
    def _resolve_from_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value


def load_settings(**overrides) -> Settings:
    """Load settings from the environment and .env; raise ConfigError on invalid values."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid configuration: {problems}") from exc
