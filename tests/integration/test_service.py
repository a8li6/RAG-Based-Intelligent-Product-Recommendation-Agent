import logging
from pathlib import Path

import pytest

from recommender.catalog import load_catalog
from recommender.config import load_settings
from recommender.grounding import GroundingError
from recommender.index import build_artifacts
from recommender.llm import LLMError
from recommender.query import InvalidQueryError, SessionState
from recommender.schemas import ExplanationPlan, Filters, RecommendationRequest
from recommender.service import create_dependencies, recommend

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"
REFERENCE = (
    "I need a laptop for Python development and travel, under $900, with at least 16 GB RAM "
    "and weight at most 1.5 kg."
)


class FakeExplainer:
    """Returns a scripted plan or raises. Counts calls."""

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.calls = 0

    def explain(self, intent, selected, evidence, catalog):
        self.calls += 1
        if isinstance(self.behavior, Exception):
            raise self.behavior
        ids = [c.product_id for c in selected]
        if self.behavior == "invented":
            ids = [*ids[:-1], "P999"]
        recs = [{"product_id": pid, "reasons": [], "unknown_fields": []} for pid in ids]
        if self.behavior == "valid":
            recs[0]["reasons"] = [
                {"code": "meets_weight_limit", "evidence_id": f"E_{ids[0]}_WEIGHT"}
            ]
        if self.behavior == "bad_evidence":
            other = ids[-1]  # a real selected product, but not recs[0]'s own
            recs[0]["reasons"] = [{"code": "catalog_fact", "evidence_id": f"E_{other}_RAM"}]
        return ExplanationPlan.model_validate({"recommendations": recs})


@pytest.fixture
def artifact_dir(tmp_path, embedder):
    return build_artifacts(load_catalog(FIXTURE), embedder, tmp_path / "catalog-v1")


def make_deps(artifact_dir, embedder, explainer=None, **overrides):
    settings = load_settings(artifact_dir=artifact_dir, **overrides)
    return create_dependencies(settings, embedder, explainer)


@pytest.fixture
def deps(artifact_dir, embedder, monkeypatch):
    from recommender.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    return make_deps(artifact_dir, embedder, explainer=None, explanation_mode="deterministic")


def ask(deps, query, session=None, **kwargs):
    return recommend(RecommendationRequest(query=query, **kwargs), session, deps)


def ids(result):
    return [r.product_id for r in result.response.recommendations]


# --- normal path ---------------------------------------------------------------------


def test_reference_request_returns_exactly_the_eligible_products(deps):
    result = ask(deps, REFERENCE)
    response = result.response
    assert response.status == "ok"
    assert set(ids(result)) == {"P001", "P004"}
    assert response.explanation_mode == "deterministic"
    assert response.clarification_question is None
    assert response.interpreted_filters.max_price_cents == 89999
    assert response.interpreted_filters.min_ram_gb == 16
    assert response.interpreted_filters.max_weight_kg == 1.5
    assert response.artifact_version == deps.artifact_version


def test_displayed_facts_come_from_the_catalog(deps):
    for item in ask(deps, REFERENCE).response.recommendations:
        product = deps.catalog.get(item.product_id)
        assert (item.title, item.price_cents, item.currency) == (
            product.title,
            product.price_cents,
            "USD",
        )
        assert item.attributes["ram_gb"] == product.ram_gb
        assert item.source_url == product.source_url
        assert item.evidence_refs


def test_ineligible_products_never_fill_the_slots(deps):
    result = ask(deps, "laptop for 3D work and gaming under $900")
    assert set(ids(result)) <= {"P001", "P003", "P004", "P006", "P008"}
    assert "P002" not in ids(result)
    assert "P005" not in ids(result)


def test_fewer_than_requested_matches_return_fewer(deps):
    result = ask(deps, "laptop for coding under $700, at least 16 GB RAM")
    assert set(ids(result)) == {"P004", "P008"}
    assert any("Only 2 product(s)" in text for text in result.response.limitations)


def test_requested_count_is_respected(deps):
    assert len(ids(ask(deps, "laptop for coding", requested_count=1))) == 1


def test_results_are_distinct_and_deterministic(deps):
    first = ids(ask(deps, "laptop for coding"))
    assert len(first) == len(set(first)) == 3
    assert ids(ask(deps, "laptop for coding")) == first


def test_limitations_disclose_snapshot_and_missing_benchmarks(deps):
    text = " ".join(
        ask(deps, "Can this run my exact CAD workload at 120 fps?").response.limitations
    )
    assert "benchmark" in text
    assert "snapshot" in text


def test_unknown_specs_are_reported_not_invented(deps):
    result = ask(deps, "laptop for desk work and coding under $700 at least 16 GB RAM")
    item = next(r for r in result.response.recommendations if r.product_id == "P008")
    assert item.attributes["weight_kg"] is None
    assert item.unknown_fields == ["weight_kg"]
    assert any("weight is not provided" in text for text in result.response.limitations)


def test_timings_cover_each_stage(deps):
    timings = ask(deps, REFERENCE).response.timings_ms
    assert set(timings) >= {"parse", "filter", "retrieve", "select", "explain", "render", "total"}
    assert all(value >= 0 for value in timings.values())


def test_request_ids_are_unique(deps):
    assert ask(deps, REFERENCE).response.request_id != ask(deps, REFERENCE).response.request_id


# --- no match, clarification, relevance ----------------------------------------------


def test_no_match_is_empty_and_explains_the_exclusions(deps):
    response = ask(deps, "laptop with at least 16 GB RAM under $300").response
    assert response.status == "no_match"
    assert response.recommendations == []
    text = " ".join(response.limitations)
    assert "budget excludes 8 of 8" in text
    assert "minimum RAM excludes 1 of 8" in text
    assert "Relaxing the budget would match" in response.clarification_question
    assert response.interpreted_filters.max_price_cents == 29999  # budget never silently changed


def test_no_match_with_unknown_weight_mentions_it(deps):
    response = ask(deps, "laptop under $700, at least 16 GB RAM, at most 1.0 kg").response
    assert response.status == "no_match"
    assert any("unknown weight" in line for line in response.limitations)


def test_ambiguous_request_asks_and_does_not_retrieve(deps, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("retrieval must not run for an ambiguous request")

    monkeypatch.setattr(deps.retriever, "search", forbid)
    response = ask(deps, "Something cheap.").response
    assert response.status == "needs_clarification"
    assert response.recommendations == []
    assert "budget" in response.clarification_question
    assert response.interpreted_filters.max_price_cents is None


def test_conflicting_controls_ask_which_to_use(deps):
    response = ask(deps, "laptop under $700", filters=Filters(max_price_cents=90000)).response
    assert response.status == "needs_clarification"
    assert "69999" in response.clarification_question
    assert "90000" in response.clarification_question


def test_structured_filters_are_applied(deps):
    result = ask(
        deps,
        "Portable laptop for coding and commuting",
        filters=Filters(max_price_cents=90000, min_ram_gb=16, max_weight_kg=1.5),
    )
    assert set(ids(result)) == {"P001", "P004"}


def test_weak_relevance_cutoff_asks_instead_of_guessing(deps, monkeypatch):
    from recommender import service

    real = service.select_candidates
    monkeypatch.setattr(service, "select_candidates", lambda c, i, k: real(c, i, k, min_score=2.0))
    response = ask(deps, "laptop for coding").response
    assert response.status == "needs_clarification"
    assert "mainly use" in response.clarification_question


@pytest.mark.parametrize("bad", ["", "   ", "x" * 2001])
def test_invalid_query_is_rejected_before_the_service(bad):
    with pytest.raises(ValueError):
        recommend(RecommendationRequest(query=bad), None, None)  # type: ignore[arg-type]


def test_whitespace_only_query_in_service(deps):
    with pytest.raises(InvalidQueryError):
        recommend(RecommendationRequest(query="\x00\x01"), None, deps)


# --- LLM and deterministic fallback --------------------------------------------------


def test_valid_llm_plan_sets_llm_mode(artifact_dir, embedder):
    explainer = FakeExplainer("valid")
    deps = make_deps(artifact_dir, embedder, explainer)
    response = ask(deps, REFERENCE).response
    assert explainer.calls == 1
    assert response.explanation_mode == "llm"
    assert response.fallback_reason is None
    first = response.recommendations[0]
    assert first.reasons[0].startswith("Weighs")
    assert not any("Basic explanation" in text for text in response.limitations)


@pytest.mark.parametrize(
    ("behavior", "reason"),
    [
        (LLMError("timeout"), "timeout"),
        (LLMError("connection_error"), "connection_error"),
        (GroundingError("malformed_json"), "malformed_json"),
        ("invented", "product_ids_mismatch"),
        ("bad_evidence", "evidence_product_mismatch"),
    ],
)
def test_llm_failure_falls_back_with_same_products(artifact_dir, embedder, behavior, reason):
    baseline = make_deps(artifact_dir, embedder, None, explanation_mode="deterministic")
    expected = ids(ask(baseline, REFERENCE))

    explainer = FakeExplainer(behavior)
    deps = make_deps(artifact_dir, embedder, explainer)
    result = ask(deps, REFERENCE)
    response = result.response
    assert explainer.calls == 1  # one attempt only
    assert response.status == "ok"
    assert response.explanation_mode == "deterministic"
    assert response.fallback_reason == reason
    assert ids(result) == expected  # an outage never changes the recommendation set
    assert any("Basic explanation" in text for text in response.limitations)
    assert all(item.reasons for item in response.recommendations)


def test_deterministic_mode_never_calls_the_llm(artifact_dir, embedder):
    explainer = FakeExplainer("valid")
    deps = make_deps(artifact_dir, embedder, explainer, explanation_mode="deterministic")
    response = ask(deps, REFERENCE).response
    assert explainer.calls == 0
    assert response.explanation_mode == "deterministic"
    assert response.fallback_reason == "llm_disabled"


def test_retriever_returning_ineligible_products_cannot_leak_them(deps, monkeypatch):
    from recommender.retrieval import Candidate

    def rogue(query, rows):
        products = deps.catalog.products
        order = ["P002", "P005", "P007", "P001", "P004"]
        return [
            Candidate(
                pid, next(i for i, p in enumerate(products) if p.product_id == pid), 1 - i / 10
            )
            for i, pid in enumerate(order)
        ]

    monkeypatch.setattr(deps.retriever, "search", rogue)
    assert ids(ask(deps, REFERENCE)) == ["P001", "P004"]


# --- sessions ------------------------------------------------------------------------


def test_budget_followup_preserves_ram_and_weight(deps):
    first = ask(deps, REFERENCE)
    assert set(ids(first)) == {"P001", "P004"}
    second = ask(deps, "under $700 instead", first.session)
    assert ids(second) == ["P004"]
    filters = second.response.interpreted_filters
    assert (filters.max_price_cents, filters.min_ram_gb, filters.max_weight_kg) == (69999, 16, 1.5)
    assert filters.semantic_query == first.response.interpreted_filters.semantic_query


def test_forget_weight_limit_widens_results(deps):
    first = ask(deps, "laptop for coding under $700, at least 16 GB RAM, at most 1.5 kg")
    assert ids(first) == ["P004"]
    second = ask(deps, "Forget the weight limit", first.session)
    assert set(ids(second)) == {"P004", "P008"}
    assert second.response.interpreted_filters.max_weight_kg is None


def test_reset_session_inherits_nothing(deps):
    ask(deps, REFERENCE)
    fresh = ask(deps, "laptop for study", SessionState())
    filters = fresh.response.interpreted_filters
    assert (filters.max_price_cents, filters.min_ram_gb, filters.max_weight_kg) == (
        None,
        None,
        None,
    )


def test_sessions_do_not_leak_between_users(deps):
    alice = ask(deps, "laptop for coding under $700").session
    bob = ask(deps, "laptop for 3D work").session
    assert alice.max_price_cents == 69999
    assert bob.max_price_cents is None
    assert ask(deps, "at least 16 GB RAM", bob).response.interpreted_filters.max_price_cents is None
    assert alice.max_price_cents == 69999


def test_clarification_leaves_the_session_unchanged(deps):
    first = ask(deps, REFERENCE)
    second = ask(deps, "laptop for $900", first.session)
    assert second.response.status == "needs_clarification"
    assert second.session == first.session


# --- logging -------------------------------------------------------------------------


def test_logs_contain_ids_and_codes_but_not_raw_user_text(deps, caplog):
    secret = "my-secret-project-name-xyz"
    with caplog.at_level(logging.INFO, logger="recommender.service"):
        ask(deps, f"laptop for {secret} under $900")
    text = " ".join(record.getMessage() for record in caplog.records)
    assert "status=ok" in text
    assert "request_id=" in text
    assert secret not in text
