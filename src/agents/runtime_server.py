"""
Lightweight IFRS9 agent runtime server.

Provides FastAPI app with:
- /healthz and /readyz for liveness/readiness
- /metrics for Prometheus scraping
- Optional /run endpoint to trigger agent-specific tasks

Security: integrates JWT auth and simple rate limiting from src.security.security_middleware.

Agent wiring:
- Configure env AGENT_NAME (e.g., orchestrator, validator, rules-engine, ml-models, integrator, reporter, data-generator, debugger)
- Optionally set AGENT_MODULE to a fully-qualified python module path to import and call.
- If AGENT_MODULE exposes a callable `main()` it will be used by the /run endpoint.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import multiprocess  # type: ignore
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from src.security.security_middleware import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    SimpleRateLimiter,
    jwt_guard,
)


logger = logging.getLogger("ifrs9.agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _load_agent_main() -> Optional[Callable[[], dict]]:
    module_path = os.getenv("AGENT_MODULE")
    if not module_path:
        # best-effort mapping by agent name
        mapping = {
            "orchestrator": "src.production_validation_generator",
            "validator": "src.validation",
            "rules-engine": "src.rules_engine",
            "ml-models": "src.enhanced_ml_models",
            "integrator": "src.gcp_integrations",
            "reporter": "src.production_validation_generator",
            "data-generator": "src.generate_data",
            "debugger": "src.ai_explanations",
        }
        agent_name = os.getenv("AGENT_NAME", "").strip().lower()
        module_path = mapping.get(agent_name)
    if not module_path:
        return None
    try:
        mod = importlib.import_module(module_path)
        candidate = getattr(mod, "main", None)
        if callable(candidate):
            return candidate  # type: ignore
        return None
    except Exception as e:
        logger.warning("Failed to import agent module %s: %s", module_path, e)
        return None


def create_app() -> FastAPI:
    app = FastAPI(title="IFRS9 Agent", version=os.getenv("APP_VERSION", "0.1.0"))

    # Security middleware
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=True)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=int(os.getenv("MAX_REQUEST_BYTES", "2000000")))
    app.add_middleware(
        SimpleRateLimiter,
        window_sec=int(os.getenv("RATE_WINDOW_SEC", "60")),
        max_requests=int(os.getenv("RATE_MAX_REQ", "120")),
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "agent": os.getenv("AGENT_NAME", "unknown")})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        # Extend to check downstream deps as needed
        return JSONResponse({"status": "ready"})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        registry = CollectorRegistry()
        try:
            multiprocess.MultiProcessCollector(registry)
        except Exception:
            pass
        data = generate_latest(registry)
        return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/run")
    async def run(_: dict | None = None, claims: dict = Depends(jwt_guard)) -> JSONResponse:
        main_fn = _load_agent_main()
        if not main_fn:
            raise HTTPException(status_code=501, detail="Agent main() not available")
        try:
            result = main_fn() or {"status": "ok"}
            return JSONResponse({"claims": claims, "result": result})
        except Exception as e:
            logger.exception("Agent run failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()

