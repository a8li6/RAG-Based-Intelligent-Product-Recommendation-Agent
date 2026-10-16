"""Catalog loading, validation, canonical snapshot and hard-constraint eligibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from recommender.schemas import Constraints, Product


@dataclass(frozen=True)
class CatalogIssue:
    line: int | None
    product_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}" if self.line is not None else "catalog"
        who = f" [{self.product_id}]" if self.product_id else ""
        return f"{where}{who}: {self.message}"


class CatalogValidationError(Exception):
    """Raised when a catalog file has any invalid record. Nothing is silently dropped."""

    def __init__(self, issues: list[CatalogIssue]):
        self.issues = issues
        super().__init__("\n".join(str(issue) for issue in issues))


class Catalog:
    """Validated, ID-sorted product snapshot. Row order is the stable vector order."""

    def __init__(self, products: list[Product]):
        self.products: list[Product] = sorted(products, key=lambda p: p.product_id)
        self._by_id = {p.product_id: p for p in self.products}

    def __len__(self) -> int:
        return len(self.products)

    def get(self, product_id: str) -> Product:
        return self._by_id[product_id]

    @property
    def product_ids(self) -> list[str]:
        return [p.product_id for p in self.products]

    def canonical_bytes(self) -> bytes:
        """Deterministic ID-sorted JSONL snapshot used for hashing and persistence."""
        lines = [
            json.dumps(
                p.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for p in self.products
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def eligible_rows(self, constraints: Constraints) -> list[int]:
        """Row positions of products satisfying every active hard constraint."""
        return [
            row for row, product in enumerate(self.products) if is_eligible(product, constraints)
        ]


def is_eligible(product: Product, constraints: Constraints) -> bool:
    """Typed predicates only. A missing value fails an active constraint."""
    if product.currency != constraints.currency:
        return False
    if constraints.require_in_stock and not product.in_stock:
        return False
    if (
        constraints.max_price_cents is not None
        and product.price_cents > constraints.max_price_cents
    ):
        return False
    if constraints.min_ram_gb is not None and (
        product.ram_gb is None or product.ram_gb < constraints.min_ram_gb
    ):
        return False
    if constraints.max_weight_kg is None:
        return True
    return product.weight_kg is not None and product.weight_kg <= constraints.max_weight_kg


def parse_catalog(text: str) -> Catalog:
    """Parse JSONL text. Collects every error and rejects the whole build if any exist."""
    issues: list[CatalogIssue] = []
    products: list[Product] = []
    seen: dict[str, int] = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(CatalogIssue(line_no, None, f"invalid JSON: {exc.msg}"))
            continue
        pid = data.get("product_id") if isinstance(data, dict) else None
        try:
            product = Product.model_validate(data)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(part) for part in err["loc"]) or "record"
                issues.append(CatalogIssue(line_no, pid, f"{loc}: {err['msg']}"))
            continue
        if product.product_id in seen:
            issues.append(
                CatalogIssue(
                    line_no,
                    product.product_id,
                    f"duplicate product_id (first seen on line {seen[product.product_id]})",
                )
            )
            continue
        seen[product.product_id] = line_no
        products.append(product)

    if issues:
        raise CatalogValidationError(issues)
    if not products:
        raise CatalogValidationError([CatalogIssue(None, None, "catalog contains no records")])
    return Catalog(products)


def load_catalog(path: str | Path) -> Catalog:
    """Read a UTF-8 JSONL catalog file."""
    return parse_catalog(Path(path).read_text(encoding="utf-8"))
