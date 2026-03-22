"""
Per-request Langfuse trace attributes via OpenTelemetry propagation.

Sets session, user, tags, and trace name for all observations created during the request
(LangChain callbacks, @observe spans, manual start_as_current_observation, etc.).

Optional headers (case-insensitive in Starlette):
- X-Langfuse-Session-Id
- X-Langfuse-User-Id
- X-Request-Id  (stored as propagated metadata key request_id)
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from langfuse import propagate_attributes


def _path_tag(path: str) -> str:
    p = path.strip("/")
    if not p:
        return "http_root"
    return p.replace("/", "_")[:200]


class LangfuseRequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        tag = _path_tag(path)
        trace_name = f"{request.method} {path}"[:200]

        session_id = (request.headers.get("x-langfuse-session-id") or "").strip() or None
        user_id = (request.headers.get("x-langfuse-user-id") or "").strip() or None
        request_id = (request.headers.get("x-request-id") or "").strip() or None

        metadata: dict[str, str] | None = None
        if request_id:
            metadata = {"request_id": request_id[:200]}

        with propagate_attributes(
            session_id=session_id[:200] if session_id else None,
            user_id=user_id[:200] if user_id else None,
            tags=[tag],
            trace_name=trace_name,
            metadata=metadata,
        ):
            return await call_next(request)
