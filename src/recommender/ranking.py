"""Candidate selection and evidence packaging.

Baseline ranking is cosine similarity, deduplicated by product ID, with the product ID as
a deterministic tie-breaker. Price or rating boosts never compensate for a failed hard
constraint. Evidence records are generated from the canonical catalog only.
"""

from __future__ import annotations

import re

from recommender.catalog import Catalog, is_eligible
from recommender.query import QueryIntent
from recommender.retrieval import Candidate
from recommender.schemas import EvidenceRecord, Product, ReasonCode

SCORE_DECIMALS = 6  # scores equal within 1e-6 are tied and ordered by product ID

EVIDENCE_SUFFIX = {
    "price_cents": "PRICE",
    "ram_gb": "RAM",
    "storage_gb": "STORAGE",
    "weight_kg": "WEIGHT",
    "gpu_type": "GPU",
    "in_stock": "STOCK",
    "use_cases": "USECASES",
}


def evidence_id(product_id: str, field: str) -> str:
    return f"E_{product_id}_{EVIDENCE_SUFFIX[field]}"


def select_candidates(
    candidates: list[Candidate],
    intent: QueryIntent,
    catalog: Catalog,
    *,
    min_score: float | None = None,
) -> list[Candidate]:
    """Unique, eligible candidates in deterministic order, at most requested_count.

    min_score is an optional abstention cutoff. It is off by default: calibrate it on a
    labeled development set, never by guessing what a cosine score means.
    """
    constraints = intent.constraints()
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        product = catalog.get(candidate.product_id)
        if not is_eligible(product, constraints):
            continue  # retrieval must never override hard eligibility
        current = best.get(candidate.product_id)
        if current is None or candidate.score > current.score:
            best[candidate.product_id] = candidate

    ordered = sorted(best.values(), key=lambda c: (-round(c.score, SCORE_DECIMALS), c.product_id))
    if min_score is not None:
        ordered = [c for c in ordered if c.score >= min_score]
    return ordered[: intent.requested_count]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def matching_use_cases(use_cases: list[str], semantic_query: str) -> list[str]:
    """Catalog use cases whose every word appears in the request (exact words, no stemming)."""
    query_tokens = _tokens(semantic_query)
    return [uc for uc in use_cases if (tokens := _tokens(uc)) and tokens <= query_tokens]


def _codes(*conditional: tuple[bool, ReasonCode]) -> tuple[ReasonCode, ...]:
    return (*(code for active, code in conditional if active), "catalog_fact")


def product_evidence(product: Product, intent: QueryIntent) -> list[EvidenceRecord]:
    """Evidence for one product. A missing spec has no allowed codes (it is 'unknown')."""
    pid = product.product_id

    def record(field: str, value, codes: tuple[ReasonCode, ...]) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id(pid, field),
            product_id=pid,
            field=field,  # type: ignore[arg-type]
            value=value,
            allowed_reason_codes=codes if value is not None else (),
        )

    budget_ok = intent.max_price_cents is not None and product.price_cents <= intent.max_price_cents
    ram_ok = (
        intent.min_ram_gb is not None
        and product.ram_gb is not None
        and product.ram_gb >= intent.min_ram_gb
    )
    weight_ok = (
        intent.max_weight_kg is not None
        and product.weight_kg is not None
        and product.weight_kg <= intent.max_weight_kg
    )
    use_ok = bool(matching_use_cases(product.use_cases, intent.semantic_query))

    return [
        record("price_cents", product.price_cents, _codes((budget_ok, "meets_budget"))),
        record("ram_gb", product.ram_gb, _codes((ram_ok, "meets_ram_minimum"))),
        record("storage_gb", product.storage_gb, _codes()),
        record("weight_kg", product.weight_kg, _codes((weight_ok, "meets_weight_limit"))),
        record("gpu_type", product.gpu_type, _codes()),
        record(
            "in_stock",
            product.in_stock,
            _codes((intent.require_in_stock and product.in_stock, "in_stock")),
        ),
        record("use_cases", list(product.use_cases), _codes((use_ok, "use_case_match"))),
    ]


def build_evidence(
    selected: list[Candidate], intent: QueryIntent, catalog: Catalog
) -> list[EvidenceRecord]:
    """Evidence records for every selected candidate, in selection order."""
    records: list[EvidenceRecord] = []
    for candidate in selected:
        records.extend(product_evidence(catalog.get(candidate.product_id), intent))
    return records
