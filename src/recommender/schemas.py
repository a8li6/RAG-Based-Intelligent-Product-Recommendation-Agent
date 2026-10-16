"""Typed contracts for catalog records."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Dataset plausibility limits; these are not claims about universal hardware limits.
MAX_PRICE_CENTS = 100_000_000
MAX_RAM_GB = 1024
MAX_STORAGE_GB = 65_536
MAX_WEIGHT_KG = 10.0


class Product(BaseModel):
    """One catalog record. Money is integer cents; unknown specs are None, never zero."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_id: str = Field(pattern=r"^P[0-9]{3,}$")
    title: str = Field(min_length=1, max_length=120)
    category: Literal["laptop"]
    brand: str = Field(min_length=1)
    price_cents: int = Field(strict=True, ge=0, le=MAX_PRICE_CENTS)
    currency: Literal["USD"]
    in_stock: bool = Field(strict=True)
    ram_gb: int | None = Field(default=None, strict=True, ge=1, le=MAX_RAM_GB)
    storage_gb: int | None = Field(default=None, strict=True, ge=1, le=MAX_STORAGE_GB)
    weight_kg: float | None = Field(default=None, gt=0, le=MAX_WEIGHT_KG, allow_inf_nan=False)
    gpu_type: Literal["integrated", "dedicated"] | None = None
    use_cases: list[str] = Field(min_length=1)
    description: str = Field(max_length=2000)
    source_url: str
    updated_at: datetime

    @field_validator("title", "brand", "description", mode="after")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return unicodedata.normalize("NFC", value)

    @field_validator("use_cases", mode="after")
    @classmethod
    def _clean_use_cases(cls, value: list[str]) -> list[str]:
        cleaned = [unicodedata.normalize("NFC", item.strip()) for item in value]
        if any(not item for item in cleaned):
            raise ValueError("use_cases must not contain empty entries")
        return cleaned

    @field_validator("source_url", mode="after")
    @classmethod
    def _check_source_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source_url must be an absolute https URL")
        return value

    @field_validator("updated_at", mode="after")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value


class Constraints(BaseModel):
    """Explicit hard constraints. A product must satisfy every active one."""

    model_config = ConfigDict(extra="forbid")

    max_price_cents: int | None = Field(default=None, strict=True, ge=0)
    currency: Literal["USD"] = "USD"
    min_ram_gb: int | None = Field(default=None, strict=True, ge=1)
    max_weight_kg: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    require_in_stock: bool = True


class Filters(BaseModel):
    """Structured UI/API controls. When supplied they are authoritative for their field."""

    model_config = ConfigDict(extra="forbid")

    max_price_cents: int | None = Field(default=None, strict=True, ge=0, le=MAX_PRICE_CENTS)
    currency: Literal["USD"] | None = None
    min_ram_gb: int | None = Field(default=None, strict=True, ge=1, le=MAX_RAM_GB)
    max_weight_kg: float | None = Field(default=None, gt=0, le=MAX_WEIGHT_KG, allow_inf_nan=False)
    require_in_stock: bool | None = None
