"""Validate an explanation plan against canonical catalog data, then render it.

The model proposes product IDs, evidence IDs and reason codes. Code decides what is
permissible. Every user-visible fact and sentence comes from the catalog or from an
approved template, never from raw model prose. Schema validity alone does not prove that
content is correct; each reference is checked against the catalog snapshot.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from recommender.catalog import Catalog, is_eligible
from recommender.query import QueryIntent
from recommender.ranking import build_evidence, matching_use_cases
from recommender.retrieval import Candidate
from recommender.schemas import (
    EvidenceRecord,
    ExplanationPlan,
    PlanReason,
    PlanRecommendation,
    Product,
    ReasonCode,
    RecommendationItem,
)

MAX_REASONS = 2
UNKNOWNABLE_FIELDS = ("ram_gb", "storage_gb", "weight_kg", "gpu_type")
FIELD_LABELS = {
    "price_cents": "price",
    "ram_gb": "RAM",
    "storage_gb": "storage",
    "weight_kg": "weight",
    "gpu_type": "GPU",
    "in_stock": "stock status",
    "use_cases": "use cases",
}
# Preference order when picking at most two reasons for the deterministic plan.
SPECIFIC_ORDER: tuple[tuple[ReasonCode, str], ...] = (
    ("use_case_match", "use_cases"),
    ("meets_budget", "price_cents"),
    ("meets_ram_minimum", "ram_gb"),
    ("meets_weight_limit", "weight_kg"),
    ("in_stock", "in_stock"),
)
FACT_ORDER = ("ram_gb", "weight_kg", "gpu_type", "storage_gb")


class GroundingError(Exception):
    """A plan failed validation. `code` is a stable reason code safe to log."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def parse_plan(raw: str | dict) -> ExplanationPlan:
    """Check JSON shape, allowed enums, bounded lists and forbidden extra fields."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise GroundingError("malformed_json", exc.msg) from exc
    try:
        return ExplanationPlan.model_validate(data)
    except ValidationError as exc:
        raise GroundingError("invalid_shape", f"{exc.error_count()} schema error(s)") from exc


def validate_plan(
    plan: ExplanationPlan,
    selected: list[Candidate],
    evidence: list[EvidenceRecord],
    intent: QueryIntent,
    catalog: Catalog,
) -> ExplanationPlan:
    """Raise GroundingError unless every claim in the plan is permissible."""
    selected_ids = [c.product_id for c in selected]
    plan_ids = [r.product_id for r in plan.recommendations]
    if len(set(plan_ids)) != len(plan_ids):
        raise GroundingError("duplicate_product", "a product appears more than once")
    if plan_ids != selected_ids:
        raise GroundingError(
            "product_ids_mismatch", "plan must match the selected products in order"
        )

    constraints = intent.constraints()
    for pid in plan_ids:
        if not is_eligible(catalog.get(pid), constraints):
            raise GroundingError("constraint_violation", pid)

    canonical = {e.evidence_id: e for e in build_evidence(selected, intent, catalog)}
    supplied = {e.evidence_id: e for e in evidence}
    if supplied != canonical:
        raise GroundingError("evidence_mismatch", "context differs from the catalog snapshot")

    for rec in plan.recommendations:
        product = catalog.get(rec.product_id)
        seen: set[str] = set()
        for reason in rec.reasons:
            record = canonical.get(reason.evidence_id)
            if record is None:
                raise GroundingError("unknown_evidence", reason.evidence_id)
            if record.product_id != rec.product_id:
                raise GroundingError("evidence_product_mismatch", reason.evidence_id)
            if reason.code not in record.allowed_reason_codes:
                raise GroundingError(
                    "reason_not_allowed", f"{reason.code} for {reason.evidence_id}"
                )
            if reason.evidence_id in seen:
                raise GroundingError("duplicate_reason", reason.evidence_id)
            seen.add(reason.evidence_id)
        for field in rec.unknown_fields:
            if getattr(product, field) is not None:
                raise GroundingError("unknown_field_claim", f"{rec.product_id}.{field} is known")
    return plan


def deterministic_plan(
    selected: list[Candidate], evidence: list[EvidenceRecord]
) -> ExplanationPlan:
    """Simple plan for the same selected products: at most two permitted reasons each."""
    by_product: dict[str, dict[str, EvidenceRecord]] = {}
    for record in evidence:
        by_product.setdefault(record.product_id, {})[record.field] = record

    recommendations = []
    for candidate in selected:
        fields = by_product[candidate.product_id]
        reasons: list[PlanReason] = []
        for code, field in SPECIFIC_ORDER:
            if len(reasons) < MAX_REASONS and code in fields[field].allowed_reason_codes:
                reasons.append(PlanReason(code=code, evidence_id=fields[field].evidence_id))
        for field in FACT_ORDER:
            record = fields[field]
            if len(reasons) < MAX_REASONS and "catalog_fact" in record.allowed_reason_codes:
                reasons.append(PlanReason(code="catalog_fact", evidence_id=record.evidence_id))
        unknown = [f for f in UNKNOWNABLE_FIELDS if fields[f].value is None]
        recommendations.append(
            PlanRecommendation(
                product_id=candidate.product_id, reasons=reasons, unknown_fields=unknown
            )
        )
    return ExplanationPlan(recommendations=recommendations)


def format_money(cents: int, currency: str = "USD") -> str:
    return f"{cents // 100}.{cents % 100:02d} {currency}"


def render_reason(reason: PlanReason, record: EvidenceRecord, intent: QueryIntent) -> str:
    """Approved sentence template filled only with canonical values and the active limits."""
    value = record.value
    if reason.code == "meets_budget":
        return (
            f"Priced at {format_money(value)}, within your maximum of "  # type: ignore[arg-type]
            f"{format_money(intent.max_price_cents)}."  # type: ignore[arg-type]
        )
    if reason.code == "meets_ram_minimum":
        return f"Has {value} GB RAM, meeting your minimum of {intent.min_ram_gb} GB."
    if reason.code == "meets_weight_limit":
        return f"Weighs {value:.2f} kg, within your limit of {intent.max_weight_kg:g} kg."  # type: ignore[str-format]
    if reason.code == "in_stock":
        return "Marked in stock in this catalog snapshot."
    if reason.code == "use_case_match":
        matches = matching_use_cases(list(value), intent.semantic_query)  # type: ignore[arg-type]
        return f"The catalog lists these matching use cases: {', '.join(matches)}."
    return f"The catalog lists {FIELD_LABELS[record.field]}: {_fact_text(record)}."


def _fact_text(record: EvidenceRecord) -> str:
    value = record.value
    if record.field in ("ram_gb", "storage_gb"):
        return f"{value} GB"
    if record.field == "weight_kg":
        return f"{value:.2f} kg"  # type: ignore[str-format]
    if record.field == "price_cents":
        return format_money(value)  # type: ignore[arg-type]
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def _attributes(product: Product) -> dict:
    return {
        "ram_gb": product.ram_gb,
        "storage_gb": product.storage_gb,
        "weight_kg": product.weight_kg,
        "gpu_type": product.gpu_type,
        "use_cases": list(product.use_cases),
    }


def render_items(
    plan: ExplanationPlan,
    evidence: list[EvidenceRecord],
    intent: QueryIntent,
    catalog: Catalog,
) -> list[RecommendationItem]:
    """Build user-visible items. Call only with a plan that passed validate_plan."""
    index = {e.evidence_id: e for e in evidence}
    items = []
    for rec in plan.recommendations:
        product = catalog.get(rec.product_id)
        refs: list[str] = []
        for reason in rec.reasons:
            ref = f"{product.product_id}.{index[reason.evidence_id].field}"
            if ref not in refs:
                refs.append(ref)
        items.append(
            RecommendationItem(
                product_id=product.product_id,
                title=product.title,
                price_cents=product.price_cents,
                currency=product.currency,
                in_stock=product.in_stock,
                source_url=product.source_url,
                attributes=_attributes(product),
                reasons=[render_reason(r, index[r.evidence_id], intent) for r in rec.reasons],
                evidence_refs=refs,
                unknown_fields=list(rec.unknown_fields),
            )
        )
    return items
