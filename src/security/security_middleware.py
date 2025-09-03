"""
Security middleware and dependencies for the IFRS9 FastAPI services.
Includes: security headers, request size limits, simple rate limiting, and JWT auth helpers.

This module is framework-light and has no external deps beyond FastAPI/Starlette when used.
It can also be imported in any Starlette-compatible ASGI app.
"""

from __future__ import annotations

import os
import time
import hmac
import base64
import json
import logging
from typing import Callable, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False, csp: Optional[str] = None) -> None:
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.csp = csp or "default-src 'self'; frame-ancestors 'none'"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("X-XSS-Protection", "0")  # modern browsers ignore/ CSP used
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", self.csp)
        if self.enable_hsts and request.url.scheme == "https":
            resp.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        return resp


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, max_bytes: int = 2_000_000) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > self.max_bytes:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            except Exception:
                pass
        return await call_next(request)


class SimpleRateLimiter(BaseHTTPMiddleware):
    """Naive in-memory IP rate limiter (per-process). For production, use Redis-backed limiter.

    window_sec: window for counting requests
    max_requests: allowed requests per window per key
    key_func: derive a key (e.g., by IP or auth subject)
    """

    def __init__(self, app: ASGIApp, *, window_sec: int = 60, max_requests: int = 120, key_func: Optional[Callable[[Request], str]] = None) -> None:
        super().__init__(app)
        self.window = window_sec
        self.max = max_requests
        self.key_func = key_func or (lambda r: r.client.host if r.client else "unknown")
        self._store: Dict[str, Dict[str, int]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        now = int(time.time())
        window_id = str(now // self.window)
        key = self.key_func(request)
        meta = self._store.setdefault(key, {})
        count = meta.get(window_id, 0)
        if count >= self.max:
            return JSONResponse({"detail": "Too Many Requests"}, status_code=429)
        meta[window_id] = count + 1
        # cleanup old windows
        for k in list(meta.keys()):
            if k != window_id:
                try:
                    if int(k) < int(window_id) - 2:
                        del meta[k]
                except Exception:
                    pass
        return await call_next(request)


# Minimal JWT validation without extra deps (supports HS256 only)
def verify_jwt_hs256(token: str, secret: str, audience: Optional[str] = None) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b = base64.urlsafe_b64decode(parts[0] + "==")
        payload_b = base64.urlsafe_b64decode(parts[1] + "==")
        sig = base64.urlsafe_b64decode(parts[2] + "==")
        header = json.loads(header_b)
        payload = json.loads(payload_b)
        if header.get("alg") != "HS256":
            return None
        data = f"{parts[0]}.{parts[1]}".encode()
        mac = hmac.new(secret.encode(), data, digestmod="sha256").digest()
        if not hmac.compare_digest(mac, sig):
            return None
        if audience and payload.get("aud") not in (audience if isinstance(audience, (list, tuple)) else [audience]):
            return None
        # exp check (optional)
        exp = payload.get("exp")
        if exp and int(time.time()) > int(exp):
            return None
        return payload
    except Exception:
        return None


async def jwt_guard(request: Request) -> Optional[dict]:
    """FastAPI dependency to enforce JWT. Returns claims on success, else 401.
    Configure with env: API_AUTH_DISABLED to bypass in dev.
    """
    from fastapi import HTTPException, status

    if os.getenv("API_AUTH_DISABLED", "false").lower() in ("1", "true", "yes"):
        return {"sub": "dev-bypass"}

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    secret = os.getenv("API_JWT_SECRET", "")
    audience = os.getenv("API_JWT_AUDIENCE")
    claims = verify_jwt_hs256(token, secret, audience)
    if not claims:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return claims

