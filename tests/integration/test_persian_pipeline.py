"""Persian queries through the real pipeline (built-in embedder, no mocks)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recommender import persian as fa
from recommender.api import create_app
from recommender.catalog import load_catalog
from recommender.documents import catalog_documents
from recommender.index import build_artifacts
from recommender.local_embedder import LocalEmbedder, concepts_of, features
from recommender.ranking import matching_use_cases
from recommender.schemas import RecommendationRequest
from recommender.service import create_dependencies, recommend

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample" / "products.jsonl"
DEMO = ROOT / "data" / "demo" / "products.jsonl"


@pytest.fixture(scope="module")
def sample():
    return load_catalog(SAMPLE)


def top(catalog, query, k=2):
    embedder = LocalEmbedder().fit(catalog_documents(catalog))
    scores = embedder.encode_products(catalog_documents(catalog)) @ embedder.encode_query(query)[0]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [catalog.products[i].product_id for i in order[:k]]


# --- features and concepts -----------------------------------------------------------------


def test_persian_words_produce_the_shared_concept_features():
    assert "concept_code" in features("برنامه‌نویسی")
    assert "concept_gaming" in features("لپ‌تاپ برای بازی")
    assert "concept_code" in features("programming")  # the English path is unchanged


def test_unknown_persian_words_produce_no_features_instead_of_noise():
    assert features("دستور پخت اسموتی موز") == {}


def test_concepts_of_merges_both_languages():
    assert concepts_of("coding برنامه‌نویسی") == frozenset({"concept_code"})
    assert concepts_of("سفر and gaming") == frozenset({"concept_travel", "concept_gaming"})


def test_persian_and_english_requests_put_the_same_product_first(sample):
    # Only the first place is compared: lower ranks on an 8-product fixture are lexical noise.
    for persian_query, english_query in (
        ("لپ‌تاپ برای بازی", "laptop for gaming"),
        ("لپ‌تاپ برای درس", "laptop for study"),
        ("لپ‌تاپ برای سفر", "laptop for travel"),
        ("لپ‌تاپ برای کدنویسی", "laptop for coding"),
    ):
        assert top(sample, persian_query, 1) == top(sample, english_query, 1), persian_query


def test_a_request_without_any_known_word_still_gives_a_valid_deterministic_vector(sample):
    assert top(sample, "نان سنگک") == top(sample, "نان سنگک")
    LocalEmbedder().fit(catalog_documents(sample)).encode_query("هیچ")  # must not raise


# --- use-case reasons ----------------------------------------------------------------------


def test_use_case_relation_works_across_languages_and_shows_catalog_values():
    uc = ["coding", "commuting", "office work", "3D work", "gaming"]
    assert matching_use_cases(uc, "برنامه‌نویسی") == ["coding"]
    assert matching_use_cases(uc, "رفت‌وآمد با مترو") == ["commuting"]
    assert matching_use_cases(uc, "رندر سه‌بعدی") == ["3D work"]
    assert matching_use_cases(uc, "لپ‌تاپ") == []


# --- service -------------------------------------------------------------------------------


@pytest.fixture
def deps(settings, tmp_path):
    cfg = settings.model_copy(update={"catalog_path": DEMO, "artifact_dir": tmp_path / "idx"})
    build_artifacts(load_catalog(DEMO), LocalEmbedder(), cfg.artifact_dir)
    return create_dependencies(cfg, LocalEmbedder())


def ask(deps, query, **kwargs):
    return recommend(RecommendationRequest(query=query, **kwargs), None, deps)


def test_persian_constraints_are_enforced_end_to_end(deps):
    result = ask(deps, "لپ‌تاپ برای برنامه‌نویسی، زیر ۹۰۰ دلار، حداقل ۱۶ گیگ رم و حداکثر ۱٫۵ کیلوگرم")
    response = result.response
    assert response.status == "ok"
    filters = response.interpreted_filters
    assert (filters.max_price_cents, filters.min_ram_gb, filters.max_weight_kg) == (89999, 16, 1.5)
    catalog = deps.catalog
    for item in response.recommendations:
        product = catalog.get(item.product_id)
        assert product.price_cents <= 89999 and product.ram_gb >= 16 and product.weight_kg <= 1.5
        assert product.in_stock
    assert response.recommendations


def test_a_toman_budget_asks_instead_of_returning_unfiltered_products(deps):
    response = ask(deps, "لپ‌تاپ برای درس زیر ۲۵ میلیون تومان").response
    assert response.status == "needs_clarification" and response.recommendations == []
    assert "USD" in response.clarification_question


def test_persian_no_match_is_empty_and_explained(deps):
    response = ask(deps, "لپ‌تاپ با حداقل ۱۲۸ گیگ رم").response
    assert response.status == "no_match" and response.recommendations == []
    assert any("minimum RAM excludes" in line for line in response.limitations)


def test_persian_follow_up_through_the_client_held_session(deps):
    first = ask(deps, "لپ‌تاپ برای کدنویسی، زیر ۱۰۰۰ دلار و حداقل ۱۶ گیگ رم").response
    second = recommend(
        RecommendationRequest(query="زیر ۷۰۰ دلار", session=first.session), None, deps
    ).response
    assert second.interpreted_filters.max_price_cents == 69999
    assert second.interpreted_filters.min_ram_gb == 16
    assert all(deps.catalog.get(i.product_id).price_cents <= 69999 for i in second.recommendations)


def test_responses_stay_english_and_use_catalog_values(deps):
    response = ask(deps, "لپ‌تاپ برای بازی زیر ۱۵۰۰ دلار").response
    assert response.status == "ok"
    for item in response.recommendations:
        assert item.title == deps.catalog.get(item.product_id).title
        assert all(not fa.has_persian(text) for text in item.reasons)
    assert all(not fa.has_persian(text) for text in response.limitations)


def test_persian_query_text_never_reaches_the_logs(deps, caplog):
    import logging

    secret = "رمزمحرمانه‌ی‌پروژه"
    with caplog.at_level(logging.INFO, logger="recommender.service"):
        ask(deps, f"لپ‌تاپ برای {secret} زیر ۹۰۰ دلار")
    assert secret not in " ".join(r.getMessage() for r in caplog.records)


# --- API -----------------------------------------------------------------------------------


def test_api_accepts_utf8_persian_and_rejects_bad_persian_values(deps):
    with TestClient(create_app(deps=deps)) as client:
        ok = client.post(
            "/recommendations",
            content=json.dumps(
                {"query": "لپ‌تاپ برای سفر زیر ۱۲۰۰ دلار"}, ensure_ascii=False
            ).encode("utf-8"),
            headers={"content-type": "application/json; charset=utf-8"},
        )
        assert ok.status_code == 200 and ok.json()["status"] == "ok"
        bad = client.post(
            "/recommendations", json={"query": "لپ‌تاپ", "filters": {"currency": "تومان"}}
        )
        assert bad.status_code == 422
        assert "تومان" not in bad.text  # the submitted value is not echoed back
