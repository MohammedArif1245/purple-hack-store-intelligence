"""
Store Intelligence — Request Logging Middleware

Generates trace_id (UUIDv4) per request. Extracts store_id from path params.
Times request execution. Logs structured JSON:
{trace_id, store_id, endpoint, method, latency_ms, status_code}
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("store_intelligence.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every request with structured JSON.
    
    Adds trace_id header to responses for debugging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Extract store_id from path if present
        store_id = self._extract_store_id(request.url.path)

        # Add trace_id to request state
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "",
                extra={
                    "trace_id": trace_id,
                    "store_id": store_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "latency_ms": round(latency_ms, 2),
                    "status_code": 500,
                    "error": str(exc),
                },
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log structured access entry
        logger.info(
            f"{request.method} {request.url.path} {response.status_code} "
            f"{latency_ms:.1f}ms",
            extra={
                "trace_id": trace_id,
                "store_id": store_id,
                "endpoint": request.url.path,
                "method": request.method,
                "latency_ms": round(latency_ms, 2),
                "status_code": response.status_code,
            },
        )

        # Attach trace_id to response for debugging
        response.headers["X-Trace-ID"] = trace_id
        return response

    @staticmethod
    def _extract_store_id(path: str) -> str | None:
        """Extract store_id from URL path like /stores/{store_id}/metrics."""
        parts = path.strip("/").split("/")
        try:
            idx = parts.index("stores")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        except ValueError:
            pass
        return None
