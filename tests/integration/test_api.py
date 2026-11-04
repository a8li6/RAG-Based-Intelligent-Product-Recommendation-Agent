import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recommender import api as api_module
from recommender.api import create_app
from recommender.catalog import load_catalog
from recommender.index import build_artifacts
from recommender.llm import LLMError
from recommender.local_embedder import LocalEmbedder
from recommender.schemas import ExplanationPlan
from recommender.service import create_dependencies

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.jsonl"
REFERENCE = {
    "query": "laptop for Python development and travel",
    "filters": {
        "max_price_cents": 90000,
        "currency": "USD",
        "min_ram_gb": 16,
        "max_weight_kg": 1.5,
        "require_in_stock": True,
    },
    "requested_count": 3,
}
SENTINEL = "zq-secret-sentinel-81723"


@pytest.fixture
def cfg(settings, tmp_path):
    return settings.model_copy(
        update={"catalog_path": FIXTURE, "artifact_dir": tmp_path / "idx", "log_level": "INFO"}
    )


@pytest.fixture
def built_cfg(cfg):
    build_artifacts(load_catalog(FIXTURE), LocalEmbedder(), cfg.artifact_dir)
    return cfg


@pytest.fixture
def client(built_cfg):
    deps = create_dependencies(built_cfg, LocalEmbedder())
    with TestClient(create_app(deps=deps)) as c:
        yield c


def ids(response):
    return [r["product_id"] for r in response.json()["recommendations"]]


# --- health and readiness ------------------------------------------------------------


def test_health_is_alive_and_loads_nothing(cfg):
    # No artifacts exist and none are built: /health must still answer.
    app = create_app(settings=cfg.model_copy(update={"auto_build_index": False}))
    assert not cfg.artifact_dir.exists()
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_ready_reports_artifact_and_mode(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["catalog_size"] == 8
    assert body["explanation_mode"] == "deterministic"
    assert body["embedding_model"] == "builtin-hashed-tfidf-v1"
    assert len(body["artifact_version"]) == 12
    assert body["llm_available"] is None


def test_ready_is_degraded_when_the_llm_is_configured_but_down(built_cfg):
    cfg = built_cfg.model_copy(
        update={"explanation_mode": "llm", "llm_base_url": "http://127.0.0.1:9"}
    )
    deps = create_dependencies(cfg, LocalEmbedder())
    with TestClient(create_app(deps=deps)) as c:
        body = c.get("/ready").json()
    assert body["status"] == "degraded"  # a deterministic fallback is still usable
    assert body["llm_available"] is False


# --- recommendations: success paths --------------------------------------------------


def test_valid_request_returns_the_public_contract(client):
    response = client.post("/recommendations", json=REFERENCE)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["explanation_mode"] == "deterministic"
    assert set(ids(response)) == {"P001", "P004"}
    assert set(body) >= {
        "request_id", "status", "explanation_mode", "interpreted_filters", "recommendations",
        "clarification_question", "limitations", "artifact_version", "session",
    }  # fmt: skip
    first = body["recommendations"][0]
    assert set(first) >= {
        "product_id", "title", "price_cents", "currency", "attributes", "reasons",
        "evidence_refs", "unknown_fields",
    }  # fmt: skip
    assert first["currency"] == "USD"


def test_free_text_only_request(client):
    response = client.post(
        "/recommendations",
        json={"query": "laptop for coding under $900, at least 16 GB RAM, at most 1.5 kg"},
    )
    assert set(ids(response)) == {"P001", "P004"}


def test_empty_result_is_a_successful_no_match(client):
    response = client.post(
        "/recommendations",
        json={"query": "laptop", "filters": {"max_price_cents": 30000, "min_ram_gb": 16}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_match"
    assert body["recommendations"] == []
    assert body["clarification_question"]


def test_ambiguous_request_is_a_successful_clarification(client):
    body = client.post("/recommendations", json={"query": "Something cheap."}).json()
    assert body["status"] == "needs_clarification"
    assert "budget" in body["clarification_question"]


def test_follow_up_uses_the_session_the_client_sent_back(client):
    first = client.post("/recommendations", json=REFERENCE).json()
    second = client.post(
        "/recommendations", json={"query": "under $700 instead", "session": first["session"]}
    )
    assert ids(second) == ["P004"]
    assert second.json()["interpreted_filters"]["min_ram_gb"] == 16


def test_the_server_keeps_no_state_between_requests(client, built_cfg):
    first = client.post("/recommendations", json=REFERENCE).json()
    follow = {"query": "under $700 instead", "session": first["session"]}
    # Same follow-up against a brand new application instance gives the same result.
    other = create_app(deps=create_dependencies(built_cfg, LocalEmbedder()))
    with TestClient(other) as c2:
        a = client.post("/recommendations", json=follow).json()
        b = c2.post("/recommendations", json=follow).json()
    assert [r["product_id"] for r in a["recommendations"]] == [
        r["product_id"] for r in b["recommendations"]
    ]
    # And a request without a session inherits nothing from earlier traffic.
    fresh = client.post("/recommendations", json={"query": "laptop for study"}).json()
    assert fresh["interpreted_filters"]["max_price_cents"] is None


# --- recommendations: validation errors ----------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "laptop", "filters": {"currency": "EUR"}},
        {"query": "laptop", "filters": {"max_price_cents": -1}},
        {"query": "laptop", "filters": {"max_price_cents": 9.5}},
        {"query": "laptop", "filters": {"min_ram_gb": 0}},
        {"query": "laptop", "filters": {"max_weight_kg": "heavy"}},
        {"query": "laptop", "filters": {"unknown": 1}},
        {"query": "laptop", "requested_count": 0},
        {"query": "laptop", "requested_count": 4},
        {"query": "laptop", "requested_count": "3"},
        {"query": "laptop", "session": {"max_price_cents": -5}},
        {"query": "laptop", "extra": True},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": 123},
        {},
    ],
)
def test_invalid_requests_are_rejected_with_a_stable_shape(client, payload):
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "invalid_request"
    assert body["details"] and all({"field", "message"} <= set(d) for d in body["details"])


def test_validation_errors_never_echo_the_submitted_values(client):
    response = client.post(
        "/recommendations", json={"query": SENTINEL, "filters": {"currency": SENTINEL}}
    )
    assert response.status_code == 422
    assert SENTINEL not in response.text


def test_control_characters_only_query_is_rejected(client):
    response = client.post("/recommendations", json={"query": "\u0000\u0001"})
    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_request"


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/recommendations", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_request"


def test_oversized_body_is_rejected_before_parsing(client):
    big = json.dumps({"query": "a" * 40_000})
    response = client.post(
        "/recommendations", content=big, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


# --- artifacts, readiness failures ---------------------------------------------------


def test_missing_artifacts_without_auto_build_is_a_controlled_503(cfg):
    app = create_app(settings=cfg.model_copy(update={"auto_build_index": False}))
    with TestClient(app) as c:
        ready = c.get("/ready")
        post = c.post("/recommendations", json={"query": "laptop"})
    for response in (ready, post):
        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "artifacts_unavailable"
        assert "build-index" in body["message"]
        assert str(cfg.artifact_dir) not in response.text  # no filesystem paths leak


def test_missing_artifacts_are_built_once_at_startup(cfg):
    assert not cfg.artifact_dir.exists()
    with TestClient(create_app(settings=cfg)) as c:
        assert c.get("/ready").status_code == 200
        assert set(ids(c.post("/recommendations", json=REFERENCE))) == {"P001", "P004"}
    assert (cfg.artifact_dir / "manifest.json").is_file()


def test_corrupt_artifacts_are_never_silently_rebuilt(built_cfg):
    vectors = built_cfg.artifact_dir / "embeddings.npy"
    original = vectors.read_bytes()
    corrupted = bytearray(original)
    corrupted[-1] ^= 0xFF  # guaranteed to differ from the original byte
    vectors.write_bytes(bytes(corrupted))
    assert vectors.read_bytes() != original
    with TestClient(create_app(settings=built_cfg)) as c:
        response = c.get("/ready")
    assert response.status_code == 503
    assert response.json()["error_code"] == "artifacts_unavailable"
    assert str(built_cfg.artifact_dir) not in response.text
    assert vectors.read_bytes() == bytes(corrupted)  # left exactly as found, not rebuilt


def test_invalid_configuration_is_a_controlled_503(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "0")
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
        response = c.get("/ready")
    assert response.status_code == 503
    assert response.json()["error_code"] == "invalid_configuration"


# --- generation fallback -------------------------------------------------------------


class FailingExplainer:
    def __init__(self, error):
        self.error = error

    def explain(self, *args):
        raise self.error


def test_llm_failure_still_returns_a_successful_deterministic_response(built_cfg):
    cfg = built_cfg.model_copy(update={"explanation_mode": "llm"})
    deps = create_dependencies(cfg, LocalEmbedder(), FailingExplainer(LLMError("timeout")))
    with TestClient(create_app(deps=deps)) as c:
        response = c.post("/recommendations", json=REFERENCE)
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["explanation_mode"] == "deterministic"
    assert body["fallback_reason"] == "timeout"
    assert set(ids(response)) == {"P001", "P004"}


def test_valid_llm_plan_is_reported_as_llm_mode(built_cfg):
    class Good:
        def explain(self, intent, selected, evidence, catalog):
            recs = [
                {"product_id": c.product_id, "reasons": [], "unknown_fields": []} for c in selected
            ]
            return ExplanationPlan.model_validate({"recommendations": recs})

    cfg = built_cfg.model_copy(update={"explanation_mode": "llm"})
    deps = create_dependencies(cfg, LocalEmbedder(), Good())
    with TestClient(create_app(deps=deps)) as c:
        assert c.post("/recommendations", json=REFERENCE).json()["explanation_mode"] == "llm"


# --- robustness ----------------------------------------------------------------------


def test_unexpected_errors_become_a_generic_500_without_details(built_cfg, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(f"secret internals {SENTINEL} /home/user/private/path")

    monkeypatch.setattr(api_module, "recommend", boom)
    deps = create_dependencies(built_cfg, LocalEmbedder())
    with TestClient(create_app(deps=deps), raise_server_exceptions=False) as c:
        response = c.post("/recommendations", json=REFERENCE)
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "internal_error"
    assert body["request_id"]
    assert SENTINEL not in response.text and "/home/user" not in response.text
    assert "Traceback" not in response.text


def test_busy_service_returns_429_instead_of_queueing_forever(built_cfg, monkeypatch):
    monkeypatch.setattr(api_module, "BUSY_WAIT_SECONDS", 0.05)
    cfg = built_cfg.model_copy(update={"max_concurrent_requests": 1})
    deps = create_dependencies(cfg, LocalEmbedder())
    app = create_app(deps=deps)
    with TestClient(app) as c:
        app.state.api_state.gate.acquire()  # simulate one request already in flight
        response = c.post("/recommendations", json=REFERENCE)
        app.state.api_state.gate.release()
        assert response.status_code == 429
        assert response.json()["error_code"] == "busy"
        assert c.post("/recommendations", json=REFERENCE).status_code == 200  # recovered


def test_the_gate_is_released_after_a_validation_style_failure(built_cfg):
    cfg = built_cfg.model_copy(update={"max_concurrent_requests": 1})
    app = create_app(deps=create_dependencies(cfg, LocalEmbedder()))
    with TestClient(app) as c:
        for _ in range(3):
            assert c.post("/recommendations", json={"query": "\u0000"}).status_code == 422
        assert c.post("/recommendations", json=REFERENCE).status_code == 200


def test_openapi_schema_is_served(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ready", "/recommendations"} <= set(paths)


def test_requests_are_logged_without_raw_text(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="recommender.service"):
        client.post("/recommendations", json={"query": f"laptop for {SENTINEL}"})
    assert SENTINEL not in " ".join(r.getMessage() for r in caplog.records)
