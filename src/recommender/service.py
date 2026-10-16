"""One service function owns the whole recommendation sequence.

The CLI and the API both call recommend(). Core logic never imports FastAPI or Streamlit.
Failure paths are part of the design: ambiguity -> clarify, zero eligible -> no_match,
weak retrieval -> clarify, model or validation failure -> deterministic explanation.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from recommender.catalog import Catalog
from recommender.config import Settings
from recommender.embeddings import Embedder
from recommender.grounding import (
    FIELD_LABELS,
    GroundingError,
    deterministic_plan,
    render_items,
    validate_plan,
)
from recommender.index import load_artifacts
from recommender.llm import Explainer, LLMError, OllamaExplainer
from recommender.query import QueryIntent, SessionState, parse_and_merge
from recommender.ranking import build_evidence, select_candidates
from recommender.retrieval import Retriever
from recommender.schemas import (
    ActiveFilters,
    Constraints,
    ExplanationMode,
    ExplanationPlan,
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger("recommender.service")

SNAPSHOT_NOTE = "Synthetic demo catalog: prices and stock are snapshot values, not live data."
BENCHMARK_NOTE = (
    "This catalog has no performance benchmarks; statements about suitability reflect "
    "catalog descriptions only."
)
BASIC_NOTE = "Basic explanation: built from verified catalog facts using fixed templates."
RELEVANCE_QUESTION = (
    "I could not find a close semantic match. What will you mainly use the laptop for?"
)


@dataclass
class Dependencies:
    retriever: Retriever
    settings: Settings
    explainer: Explainer | None = None

    @property
    def catalog(self) -> Catalog:
        return self.retriever.artifacts.catalog

    @property
    def artifact_version(self) -> str:
        return self.retriever.artifacts.version


@dataclass
class ServiceResult:
    response: RecommendationResponse
    session: SessionState


def create_dependencies(
    settings: Settings, embedder: Embedder, explainer: Explainer | None = None
) -> Dependencies:
    """Load verified artifacts once. The LLM explainer is created only in llm mode."""
    artifacts = load_artifacts(settings.artifact_dir)
    retriever = Retriever(artifacts, embedder, settings.retrieval_k)
    if explainer is None and settings.explanation_mode == "llm":
        explainer = OllamaExplainer(settings)
    return Dependencies(retriever, settings, explainer)


def _active_filters(intent: QueryIntent) -> ActiveFilters:
    return ActiveFilters(
        semantic_query=intent.semantic_query,
        max_price_cents=intent.max_price_cents,
        currency="USD",
        min_ram_gb=intent.min_ram_gb,
        max_weight_kg=intent.max_weight_kg,
        require_in_stock=intent.require_in_stock,
    )


def _single_constraints(intent: QueryIntent) -> dict[str, Constraints]:
    """Each active constraint alone, labeled for exclusion explanations."""
    alone: dict[str, Constraints] = {}
    if intent.max_price_cents is not None:
        alone["budget"] = Constraints(
            max_price_cents=intent.max_price_cents, require_in_stock=False
        )
    if intent.min_ram_gb is not None:
        alone["minimum RAM"] = Constraints(min_ram_gb=intent.min_ram_gb, require_in_stock=False)
    if intent.max_weight_kg is not None:
        alone["maximum weight"] = Constraints(
            max_weight_kg=intent.max_weight_kg, require_in_stock=False
        )
    if intent.require_in_stock:
        alone["in-stock requirement"] = Constraints(require_in_stock=True)
    return alone


def explain_no_match(catalog: Catalog, intent: QueryIntent) -> tuple[list[str], str]:
    """Say which explicit constraints excluded products and offer a relaxation as a question."""
    total = len(catalog)
    full = intent.constraints()
    lines: list[str] = []
    relax: list[tuple[int, str]] = []

    for label, only in _single_constraints(intent).items():
        excluded = total - len(catalog.eligible_rows(only))
        lines.append(f"The {label} excludes {excluded} of {total} products.")
        loosened = full.model_copy(update=_neutral(label))
        relax.append((len(catalog.eligible_rows(loosened)), label))
    if intent.max_weight_kg is not None:
        lines.append("Products with an unknown weight cannot satisfy a weight limit.")

    best_count, best_label = max(relax, key=lambda item: (item[0], -relax.index(item)))
    if best_count > 0:
        question = (
            f"No product meets all of your constraints. Relaxing the {best_label} would match "
            f"{best_count} product(s). Would you like to try that?"
        )
    else:
        question = (
            "No product meets all of your constraints, even with any single one relaxed. "
            "Would you like to loosen more than one?"
        )
    return lines, question


def _neutral(label: str) -> dict:
    return {
        "budget": {"max_price_cents": None},
        "minimum RAM": {"min_ram_gb": None},
        "maximum weight": {"max_weight_kg": None},
        "in-stock requirement": {"require_in_stock": False},
    }[label]


def recommend(
    request: RecommendationRequest,
    session: SessionState | None,
    deps: Dependencies,
) -> ServiceResult:
    """Run the full workflow for one request. Raises InvalidQueryError on empty/oversized text."""
    settings = deps.settings
    catalog = deps.catalog
    request_id = uuid.uuid4().hex
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - since) * 1000, 3)
        return now

    def respond(
        status, intent, items=(), question=None, limitations=(), mode="deterministic", reason=None
    ) -> RecommendationResponse:
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        return RecommendationResponse(
            request_id=request_id,
            status=status,
            explanation_mode=mode,
            interpreted_filters=_active_filters(intent),
            recommendations=list(items),
            clarification_question=question,
            limitations=list(limitations),
            artifact_version=deps.artifact_version,
            fallback_reason=reason,
            timings_ms=timings,
        )

    parsed = parse_and_merge(
        request.query,
        request.filters,
        session,
        requested_count=min(request.requested_count, settings.max_recommendations),
        max_chars=settings.max_query_chars,
    )
    intent, new_session = parsed.intent, parsed.session
    t = lap("parse", started)

    if intent.needs_clarification:
        response = respond("needs_clarification", intent, question=intent.clarification_question)
        return _finish(response, new_session, 0, 0, [])

    eligible = deps.retriever.eligible_rows(intent.constraints())
    t = lap("filter", t)
    if not eligible:
        lines, question = explain_no_match(catalog, intent)
        response = respond(
            "no_match", intent, question=question, limitations=[*lines, SNAPSHOT_NOTE]
        )
        return _finish(response, new_session, 0, 0, [])

    candidates = deps.retriever.search(intent.semantic_query, eligible)
    t = lap("retrieve", t)
    selected = select_candidates(candidates, intent, catalog)
    t = lap("select", t)
    if not selected:
        response = respond("needs_clarification", intent, question=RELEVANCE_QUESTION)
        return _finish(response, new_session, len(eligible), len(candidates), [])

    evidence = build_evidence(selected, intent, catalog)
    plan, mode, reason = _explain_or_fallback(deps, intent, selected, evidence)
    t = lap("explain", t)
    items = render_items(plan, evidence, intent, catalog)
    lap("render", t)

    limitations = [SNAPSHOT_NOTE, BENCHMARK_NOTE]
    if mode == "deterministic":
        limitations.append(BASIC_NOTE)
    if len(items) < intent.requested_count:
        limitations.append(
            f"Only {len(items)} product(s) satisfy all of your constraints, so fewer than "
            f"{intent.requested_count} are shown."
        )
    for item in items:
        for field in item.unknown_fields:
            limitations.append(
                f"{item.title}: {FIELD_LABELS[field]} is not provided by this catalog."
            )

    response = respond("ok", intent, items, limitations=limitations, mode=mode, reason=reason)
    return _finish(
        response, new_session, len(eligible), len(candidates), [i.product_id for i in items]
    )


def _explain_or_fallback(
    deps: Dependencies, intent: QueryIntent, selected, evidence
) -> tuple[ExplanationPlan, ExplanationMode, str | None]:
    """One LLM attempt. Any failure or invalid plan falls back for the same selected products."""
    catalog = deps.catalog
    reason: str | None = None
    if deps.settings.explanation_mode == "llm" and deps.explainer is not None:
        try:
            plan = deps.explainer.explain(intent, selected, evidence, catalog)
            plan = validate_plan(plan, selected, evidence, intent, catalog)
            return plan, "llm", None
        except (LLMError, GroundingError) as exc:
            reason = exc.code
    else:
        reason = "llm_disabled"
    plan = deterministic_plan(selected, evidence)
    return validate_plan(plan, selected, evidence, intent, catalog), "deterministic", reason


def _finish(
    response: RecommendationResponse,
    session: SessionState,
    eligible_count: int,
    candidate_count: int,
    selected_ids: list[str],
) -> ServiceResult:
    # Operational log: IDs, counts, durations and bounded codes only; never raw user text.
    logger.info(
        "request_id=%s status=%s mode=%s artifact=%s eligible=%d candidates=%d selected=%s "
        "fallback=%s timings_ms=%s",
        response.request_id,
        response.status,
        response.explanation_mode,
        response.artifact_version,
        eligible_count,
        candidate_count,
        ",".join(selected_ids) or "-",
        response.fallback_reason or "-",
        response.timings_ms,
    )
    return ServiceResult(response=response, session=session)
