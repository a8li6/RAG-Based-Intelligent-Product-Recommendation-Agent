"""FastAPI application. Thin: validation, readiness and error mapping around recommend().

The first API is stateless (the client holds the follow-up session). Bind it to a local
interface for the demo; a public deployment needs its own authentication, rate limiting and
TLS design. Responses never contain stack traces, file paths or echoed user input.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from recommender import __version__
from recommender.catalog import CatalogValidationError, load_catalog
from recommender.config import ConfigError, Settings, load_settings
from recommender.factory import default_embedder_factory
from recommender.index import ArtifactError, build_artifacts
from recommender.llm import is_available
from recommender.query import InvalidQueryError
from recommender.schemas import RecommendationRequest, RecommendationResponse
from recommender.service import Dependencies, create_dependencies, recommend

logger = logging.getLogger("recommender.api")

DEFAULT_MAX_BODY = 16_384
BUSY_WAIT_SECONDS = 2.0


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


@dataclass
class _State:
    deps: Dependencies | None = None
    settings: Settings | None = None
    error_code: str | None = None
    error_message: str | None = None
    gate: threading.BoundedSemaphore | None = None


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    body = ErrorResponse(error_code=code, message=message, **extra)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def _startup(state: _State, settings: Settings | None) -> None:
    """Load settings, artifacts and the embedder once per process."""
    try:
        settings = settings or load_settings()
    except ConfigError:
        state.error_code, state.error_message = "invalid_configuration", "Configuration is invalid."
        logger.error("startup failed: invalid configuration")
        return
    state.settings = settings
    state.gate = threading.BoundedSemaphore(settings.max_concurrent_requests)
    try:
        embedder = default_embedder_factory(settings)
        if not settings.artifact_dir.exists() and settings.auto_build_index:
            catalog = load_catalog(settings.catalog_path)
            build_artifacts(catalog, embedder, settings.artifact_dir)
            logger.info("built missing index artifacts at startup (%d products)", len(catalog))
        state.deps = create_dependencies(settings, embedder)
    except (ArtifactError, CatalogValidationError, OSError, ValueError, ImportError) as exc:
        state.error_code = "artifacts_unavailable"
        state.error_message = (
            "Index artifacts are missing or invalid. Rebuild them with the build-index command."
        )
        logger.error("startup failed: %s", type(exc).__name__)  # type only; no paths or text


def create_app(deps: Dependencies | None = None, settings: Settings | None = None) -> FastAPI:
    """Application factory. Pass `deps` to inject loaded dependencies (tests, embedding)."""
    state = _State()
    if deps is not None:
        state.deps = deps
        state.settings = deps.settings
        state.gate = threading.BoundedSemaphore(deps.settings.max_concurrent_requests)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.deps is None and state.error_code is None:
            _startup(state, settings)
        yield

    app = FastAPI(
        title="RAG-Based Intelligent Product Recommendation Agent",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.api_state = state

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        limit = state.settings.max_request_bytes if state.settings else DEFAULT_MAX_BODY
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            return _error(413, "payload_too_large", "The request body is too large.")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Field paths and messages only: never echo the submitted values back.
        details = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
            for err in exc.errors()
        ]
        return _error(422, "invalid_request", "The request is not valid.", details=details)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        request_id = uuid.uuid4().hex
        logger.error("unhandled error request_id=%s type=%s", request_id, type(exc).__name__)
        return _error(500, "internal_error", "An internal error occurred.", request_id=request_id)

    @app.get("/health")
    def health() -> dict:
        """Process liveness only. Loads nothing."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if state.deps is None:
            return _error(
                503,
                state.error_code or "starting",
                state.error_message or "The service is starting.",
            )
        settings = state.deps.settings
        llm_wanted = settings.explanation_mode == "llm"
        llm_up = is_available(settings) if llm_wanted else None
        degraded = llm_wanted and not llm_up  # a usable deterministic fallback remains
        return {
            "status": "degraded" if degraded else "ready",
            "artifact_version": state.deps.artifact_version,
            "catalog_size": len(state.deps.catalog),
            "explanation_mode": settings.explanation_mode,
            "llm_available": llm_up,
            "embedding_model": state.deps.retriever.artifacts.manifest["embedding_model"],
        }

    @app.post("/recommendations", response_model=RecommendationResponse)
    def recommendations(request: RecommendationRequest):
        if state.deps is None or state.gate is None:
            return _error(
                503,
                state.error_code or "starting",
                state.error_message or "The service is starting.",
            )
        if not state.gate.acquire(timeout=BUSY_WAIT_SECONDS):
            return _error(429, "busy", "The service is busy. Please retry shortly.")
        try:
            result = recommend(request, request.session, state.deps)
        except InvalidQueryError:
            return _error(422, "invalid_request", "The query is empty or too long.")
        finally:
            state.gate.release()
        return result.response

    return app


app = create_app()
