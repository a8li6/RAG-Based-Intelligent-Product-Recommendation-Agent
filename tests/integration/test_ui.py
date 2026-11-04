"""Run the real Streamlit script with Streamlit's AppTest against an in-process API."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from recommender import client as client_module
from recommender.api import create_app
from recommender.catalog import load_catalog
from recommender.index import build_artifacts
from recommender.local_embedder import LocalEmbedder
from recommender.service import create_dependencies

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "sample" / "products.jsonl"
UI = str(ROOT / "ui" / "app.py")
LABEL = "Synthetic demo catalog / snapshot prices"


@pytest.fixture
def api(settings, tmp_path, monkeypatch):
    cfg = settings.model_copy(update={"catalog_path": FIXTURE, "artifact_dir": tmp_path / "idx"})
    build_artifacts(load_catalog(FIXTURE), LocalEmbedder(), cfg.artifact_dir)
    test_client = TestClient(create_app(deps=create_dependencies(cfg, LocalEmbedder())))
    test_client.__enter__()
    monkeypatch.setattr(
        client_module,
        "make_client",
        lambda base_url=client_module.DEFAULT_BASE_URL: client_module.RecommenderClient(
            "http://testserver", http=test_client
        ),
    )
    yield test_client
    test_client.__exit__(None, None, None)


def new_app() -> AppTest:
    return AppTest.from_file(UI, default_timeout=30).run()


def submit(at: AppTest, query: str, **controls) -> AppTest:
    at.text_input(key="query").input(query)
    for key, value in controls.items():
        widget = at.toggle(key=key) if key == "in_stock" else at.number_input(key=key)
        widget.set_value(value)
    button = next(b for b in at.button if b.label == "Get recommendations")
    return button.click().run()


def texts(at: AppTest) -> str:
    parts = [e.value for e in at.text] + [e.value for e in at.caption]
    parts += (
        [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.error]
    )
    parts += [e.value for e in at.markdown] + [e.value for e in at.subheader]
    return "\n".join(parts)


def titles(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader]


# --- first render --------------------------------------------------------------------


def test_first_render_shows_the_synthetic_label_and_no_results(api):
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "RAG-Based Intelligent Product Recommendation Agent"
    assert LABEL in at.info[0].value
    assert titles(at) == []
    assert "Example:" in texts(at)
    assert "ready" in texts(at) and "8 products" in texts(at)


# --- recommendations -----------------------------------------------------------------


def test_reference_request_shows_cards_filters_and_evidence(api):
    at = submit(new_app(), "Portable laptop for coding and commuting",
                budget=900.0, min_ram=16, max_weight=1.5)  # fmt: skip
    assert not at.exception
    assert set(titles(at)) == {"Atlas Air 14", "Commuter 13"}
    body = texts(at)
    assert "Budget up to 900.00 USD" in body
    assert "RAM at least 16 GB" in body
    assert "Weight at most 1.5 kg" in body
    assert "Basic explanation" in body
    assert "USD" in body  # currency is explicit next to every amount
    assert "not a purchase link" in body
    assert "Evidence fields:" in body
    assert [e.label for e in at.expander if "Evidence" in e.label]  # expandable evidence panel


def test_unknown_specification_is_displayed_as_unknown_not_zero(api):
    at = submit(new_app(), "laptop for desk work and coding", budget=700.0, min_ram=16)
    body = texts(at)
    assert "DeskMate 15" in titles(at)
    assert "Weight: not provided by this catalog" in body
    assert "Weight: 0" not in body


def test_follow_up_keeps_other_filters_through_the_client_held_session(api):
    at = submit(
        new_app(), "laptop for coding and commuting", budget=900.0, min_ram=16, max_weight=1.5
    )
    assert set(titles(at)) == {"Atlas Air 14", "Commuter 13"}
    at.number_input(key="budget").set_value(None)  # the typed follow-up now sets the budget
    at = submit(at, "under $700 instead")
    assert titles(at) == ["Commuter 13"]
    body = texts(at)
    assert "Budget up to 699.99 USD" in body
    assert "RAM at least 16 GB" in body  # preserved from the earlier turn


# --- clarification, no match, conflicts ----------------------------------------------


def test_ambiguous_request_asks_a_question(api):
    at = submit(new_app(), "Something cheap.")
    assert titles(at) == []
    assert "maximum budget" in at.info[-1].value


def test_no_match_is_explained_and_never_silently_relaxed(api):
    at = submit(new_app(), "laptop with at least 16 GB RAM under $300")
    assert titles(at) == []
    assert at.warning and "No catalog product meets all" in at.warning[-1].value
    body = texts(at)
    assert "excludes" in body
    assert "Relaxing the budget" in at.info[-1].value
    assert "Budget up to 299.99 USD" in body  # the budget stayed as requested


def test_conflict_between_control_and_text_shows_both_values(api):
    at = submit(new_app(), "laptop under $700", budget=900.0)
    assert titles(at) == []
    question = at.info[-1].value
    assert "69999" in question and "90000" in question


# --- failures and validation ---------------------------------------------------------


def test_invalid_control_value_is_rejected_before_any_request(api, monkeypatch):
    calls = []
    real = client_module.RecommenderClient.recommend
    monkeypatch.setattr(
        client_module.RecommenderClient,
        "recommend",
        lambda self, r: calls.append(r) or real(self, r),
    )
    at = submit(new_app(), "laptop", budget=899.999)
    assert any("two decimal places" in e.value for e in at.error)
    assert calls == []


def test_empty_query_asks_for_input(api):
    at = submit(new_app(), "   ")
    assert any("describe what you need" in e.value for e in at.error)


def test_api_unreachable_is_understandable_without_terminal_output(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    dead = client_module.RecommenderClient(
        "http://127.0.0.1:1", http=httpx.Client(transport=httpx.MockTransport(refuse))
    )
    monkeypatch.setattr(client_module, "make_client", lambda base_url=None: dead)
    at = new_app()
    assert any("not reachable or not ready" in w.value for w in at.warning)
    at = submit(at, "laptop for coding")
    assert not at.exception
    message = at.error[-1].value
    assert "not reachable" in message
    assert "uvicorn recommender.api:app" in message  # how to recover


def test_api_validation_error_is_shown_in_plain_language(api, monkeypatch):
    monkeypatch.setattr(
        client_module.RecommenderClient,
        "recommend",
        lambda self, r: (_ for _ in ()).throw(
            client_module.ClientError("busy", "The service is busy. Please try again in a moment.")
        ),
    )
    at = submit(new_app(), "laptop for coding")
    assert "busy" in at.error[-1].value


# --- reset ---------------------------------------------------------------------------


def test_reset_clears_every_constraint_and_the_result(api):
    at = submit(new_app(), "laptop for coding", budget=900.0, min_ram=16, max_weight=1.5)
    assert titles(at)
    assert at.session_state["session"] is not None
    reset = next(b for b in at.button if b.label == "Reset conversation")
    at = reset.click().run()
    assert titles(at) == []
    assert at.session_state["session"] is None
    assert at.session_state["response"] is None
    assert at.number_input(key="budget").value is None
    assert at.number_input(key="min_ram").value is None
    assert at.number_input(key="max_weight").value is None
    assert at.toggle(key="in_stock").value is True
    assert at.text_input(key="query").value == ""
    # A new request after reset inherits nothing.
    at = submit(at, "laptop for study")
    assert "Budget up to" not in texts(at)
    assert "RAM at least" not in texts(at)
