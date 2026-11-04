"""HTTP client and input helpers for the UI. The UI talks to the API only through this module."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import httpx
from pydantic import ValidationError

from recommender.schemas import Filters, RecommendationRequest, RecommendationResponse

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 15.0


class ClientError(Exception):
    """A failure the UI can show as a plain message. `kind` is a stable category."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)


class RecommenderClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = http  # injectable for tests (for example an ASGI-backed client)

    def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            if self._http is not None:
                return self._http.request(
                    method, self.base_url + path, timeout=self.timeout, **kwargs
                )
            with httpx.Client(timeout=self.timeout) as http:
                return http.request(method, self.base_url + path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ClientError(
                "timeout", "The service took too long to answer. Please try again."
            ) from exc
        except httpx.HTTPError as exc:
            raise ClientError(
                "unavailable", "The recommendation service is not reachable. Is the API running?"
            ) from exc

    def ready(self) -> dict | None:
        """Readiness payload, or None when the service is not ready or not reachable."""
        try:
            response = self._send("GET", "/ready")
        except ClientError:
            return None
        return response.json() if response.status_code == 200 else None

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        response = self._send("POST", "/recommendations", json=request.model_dump(mode="json"))
        if response.status_code == 200:
            try:
                return RecommendationResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise ClientError(
                    "bad_response", "The service returned an unexpected response."
                ) from exc
        raise _error_for(response)


def _error_for(response: httpx.Response) -> ClientError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    code = body.get("error_code") if isinstance(body, dict) else None
    message = body.get("message", "") if isinstance(body, dict) else ""
    status = response.status_code
    if status == 422:
        fields = [d.get("field", "") for d in body.get("details", []) if isinstance(d, dict)]
        shown = ", ".join(f for f in fields if f) or "the request"
        return ClientError("invalid_request", f"Please check {shown}: the value is not valid.")
    if status == 429:
        return ClientError("busy", "The service is busy. Please try again in a moment.")
    if status == 413:
        return ClientError("invalid_request", "The request is too large.")
    if status == 503:
        return ClientError("unavailable", message or "The service is not ready yet.")
    if status >= 500:
        return ClientError("server_error", "The service had an internal problem. Please try again.")
    return ClientError(code or "error", message or f"Unexpected response ({status}).")


def build_filters(
    budget_usd: float | None,
    min_ram_gb: float | None,
    max_weight_kg: float | None,
    in_stock_only: bool,
) -> Filters:
    """Validate UI control values before they are sent. Empty (None) means 'no limit'.

    Raises ValueError with a message the UI can show. The budget is converted to whole cents
    with Decimal; more than two decimal places is rejected instead of rounded.
    """
    max_price_cents = None
    if budget_usd is not None:
        try:
            cents = Decimal(str(budget_usd)) * 100
        except InvalidOperation as exc:
            raise ValueError("The budget is not a number.") from exc
        if cents != cents.to_integral_value():
            raise ValueError("The budget can have at most two decimal places.")
        max_price_cents = int(cents)

    ram = None
    if min_ram_gb is not None:
        if float(min_ram_gb) != int(min_ram_gb):
            raise ValueError("Minimum RAM must be a whole number of GB.")
        ram = int(min_ram_gb)

    try:
        return Filters(
            max_price_cents=max_price_cents,
            min_ram_gb=ram,
            max_weight_kg=max_weight_kg,
            require_in_stock=bool(in_stock_only),
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        field = {"max_price_cents": "budget", "min_ram_gb": "minimum RAM",
                 "max_weight_kg": "maximum weight"}.get(str(first["loc"][0]), str(first["loc"][0]))  # fmt: skip
        raise ValueError(f"The {field} is not valid.") from exc


def format_money(cents: int, currency: str = "USD") -> str:
    return f"{cents // 100}.{cents % 100:02d} {currency}"


def describe_filters(filters: dict) -> list[str]:
    """Human-readable active filters, always with the currency next to money."""
    parts = []
    if filters.get("max_price_cents") is not None:
        parts.append(
            f"Budget up to {format_money(filters['max_price_cents'], filters['currency'])}"
        )
    if filters.get("min_ram_gb") is not None:
        parts.append(f"RAM at least {filters['min_ram_gb']} GB")
    if filters.get("max_weight_kg") is not None:
        parts.append(f"Weight at most {filters['max_weight_kg']:g} kg")
    parts.append(
        "In stock only" if filters.get("require_in_stock") else "Out-of-stock items included"
    )
    return parts
