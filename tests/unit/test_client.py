import httpx
import pytest

from recommender.client import (
    ClientError,
    RecommenderClient,
    build_filters,
    describe_filters,
    format_money,
)
from recommender.schemas import RecommendationRequest


def client_with(handler):
    return RecommenderClient(
        "http://api.test", http=httpx.Client(transport=httpx.MockTransport(handler))
    )


# --- build_filters -------------------------------------------------------------------


def test_empty_controls_mean_no_limits():
    filters = build_filters(None, None, None, True)
    assert (filters.max_price_cents, filters.min_ram_gb, filters.max_weight_kg) == (
        None,
        None,
        None,
    )
    assert filters.require_in_stock is True


@pytest.mark.parametrize(
    ("budget", "cents"),
    [(900, 90000), (899.99, 89999), (0, 0), (1234.5, 123450), (0.07, 7), (19.99, 1999)],
)
def test_budget_is_converted_with_decimal_not_float_rounding(budget, cents):
    assert build_filters(budget, None, None, True).max_price_cents == cents


@pytest.mark.parametrize("bad", [899.999, 0.001, -5, float("nan")])
def test_bad_budget_is_rejected(bad):
    with pytest.raises(ValueError):
        build_filters(bad, None, None, True)


@pytest.mark.parametrize(
    ("ram", "ok"), [(16, True), (16.0, True), (16.5, False), (0, False), (-8, False)]
)
def test_minimum_ram_must_be_a_positive_whole_number(ram, ok):
    if ok:
        assert build_filters(None, ram, None, True).min_ram_gb == int(ram)
    else:
        with pytest.raises(ValueError):
            build_filters(None, ram, None, True)


@pytest.mark.parametrize("weight", [0, -1.0, float("inf")])
def test_bad_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="maximum weight"):
        build_filters(None, None, weight, True)


def test_validation_messages_name_the_field_in_plain_words():
    with pytest.raises(ValueError, match="budget"):
        build_filters(-1, None, None, True)


# --- display helpers -----------------------------------------------------------------


def test_money_always_shows_the_currency():
    assert format_money(84900) == "849.00 USD"
    assert format_money(5, "USD") == "0.05 USD"


def test_active_filters_are_described_with_units():
    parts = describe_filters(
        {"max_price_cents": 89999, "currency": "USD", "min_ram_gb": 16, "max_weight_kg": 1.5,
         "require_in_stock": True}
    )  # fmt: skip
    assert parts == [
        "Budget up to 899.99 USD",
        "RAM at least 16 GB",
        "Weight at most 1.5 kg",
        "In stock only",
    ]
    assert describe_filters({"currency": "USD", "require_in_stock": False}) == [
        "Out-of-stock items included"
    ]


# --- error mapping -------------------------------------------------------------------


def request():
    return RecommendationRequest(query="laptop")


def test_connection_failure_is_reported_as_unavailable():
    def refuse(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ClientError) as err:
        client_with(refuse).recommend(request())
    assert err.value.kind == "unavailable"
    assert "not reachable" in err.value.message


def test_timeout_is_reported_as_timeout():
    def slow(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ClientError) as err:
        client_with(slow).recommend(request())
    assert err.value.kind == "timeout"


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (
            422,
            {
                "error_code": "invalid_request",
                "details": [{"field": "filters.currency", "message": "x"}],
            },
            "invalid_request",
        ),
        (413, {"error_code": "payload_too_large"}, "invalid_request"),
        (429, {"error_code": "busy"}, "busy"),
        (
            503,
            {"error_code": "artifacts_unavailable", "message": "Rebuild the index."},
            "unavailable",
        ),
        (500, {"error_code": "internal_error", "request_id": "abc"}, "server_error"),
        (502, {}, "server_error"),
    ],
)
def test_http_errors_map_to_ui_friendly_kinds(status, body, kind):
    with pytest.raises(ClientError) as err:
        client_with(lambda r: httpx.Response(status, json=body)).recommend(request())
    assert err.value.kind == kind
    assert "Traceback" not in err.value.message


def test_validation_error_message_names_the_field_not_the_value():
    body = {
        "error_code": "invalid_request",
        "details": [{"field": "filters.currency", "message": "m"}],
    }
    with pytest.raises(ClientError) as err:
        client_with(lambda r: httpx.Response(422, json=body)).recommend(request())
    assert "filters.currency" in err.value.message


def test_non_json_and_wrong_shape_responses_are_bad_responses():
    for response in (httpx.Response(200, text="<html>"), httpx.Response(200, json={"x": 1})):
        with pytest.raises(ClientError) as err:
            client_with(lambda r, resp=response: resp).recommend(request())
        assert err.value.kind == "bad_response"


def test_ready_returns_payload_or_none():
    assert client_with(lambda r: httpx.Response(200, json={"status": "ready"})).ready() == {
        "status": "ready"
    }
    assert client_with(lambda r: httpx.Response(503, json={})).ready() is None

    def refuse(req):
        raise httpx.ConnectError("refused")

    assert client_with(refuse).ready() is None


def test_request_is_sent_as_json_to_the_recommendations_endpoint():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), req.content
        return httpx.Response(503, json={})

    with pytest.raises(ClientError):
        client_with(handler).recommend(RecommendationRequest(query="laptop for study"))
    assert seen["url"] == "http://api.test/recommendations"
    assert b"laptop for study" in seen["body"]
