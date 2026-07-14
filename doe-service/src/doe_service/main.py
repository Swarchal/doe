"""Application factory.

Run locally with ``uvicorn --factory doe_service.main:create_app``.
"""

from importlib.metadata import version
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from doe_service import __version__
from doe_service.errors import error_envelope, register_exception_handlers
from doe_service.limits import DEFAULT_LIMITS, Limits
from doe_service.routers import analysis, designs, optimize, plot_data

# Minimal ASGI aliases -- the middleware below is pure ASGI (see BodySizeLimitMiddleware),
# so it speaks in scope/receive/send rather than Request/Response.
Scope = dict[str, Any]
Message = dict[str, Any]


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_body_bytes`` -- by *counting bytes*, not by trusting
    ``Content-Length``.

    A ``Content-Length`` check alone is only an advisory one: a client that sends
    ``Transfer-Encoding: chunked``, or simply omits the header, declares no length at all and
    sails past it, and nothing further down the stack bounds what gets read into memory. So
    the header is still checked first (it lets an oversized request be refused before a single
    byte of body arrives), but the receive channel is then wrapped to tally the chunks actually
    delivered and abort the moment the tally crosses the cap.

    Pure ASGI rather than ``@app.middleware("http")``: the Starlette ``BaseHTTPMiddleware``
    that decorator installs hands the endpoint its own ``Request``, giving no seam to wrap the
    receive channel through.
    """

    def __init__(self, app: Any, *, limits: Limits) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cap = self.limits.max_body_bytes
        declared = self._declared_length(scope)
        if declared is not None and declared > cap:
            await self._too_large(scope, receive, send, declared)
            return

        # Drain the body here, counting as we go, and stop the moment the tally passes the cap
        # -- so an unbounded chunked stream is abandoned after at most `cap` bytes rather than
        # read into memory in full. The buffered body is then replayed to the app below.
        # (Raising out of a wrapped receive() instead would not work: FastAPI catches whatever
        # the body read throws and reports it as a 400 "error parsing the body", losing the
        # 413 envelope.)
        body = bytearray()
        while True:
            message: Message = await receive()
            if message["type"] != "http.request":
                break  # http.disconnect: client went away, let the app see it
            body.extend(message.get("body", b""))
            if len(body) > cap:
                await self._too_large(scope, receive, send, len(body))
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if replayed:
                trailing: Message = await receive()
                return trailing
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _declared_length(scope: Scope) -> int | None:
        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        for name, value in headers:
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def _too_large(self, scope: Scope, receive: Any, send: Any, size: int) -> None:
        response = JSONResponse(
            status_code=413,
            content=error_envelope(
                "limit_exceeded",
                f"request body of {size} bytes exceeds the cap of "
                f"{self.limits.max_body_bytes} bytes",
            ),
        )
        await response(scope, receive, send)


def create_app(*, limits: Limits = DEFAULT_LIMITS) -> FastAPI:
    """Build the FastAPI application with all routers mounted.

    ``limits`` is stashed on ``app.state`` (``docs/WEBSERVICE_API.md`` "Limits": "All
    configurable at deployment") and read by :class:`BodySizeLimitMiddleware`, the one
    check that has to run ahead of routing -- an oversized body never reaches a router, so
    it cannot go through the usual ``LimitExceeded`` -> 422 path; per the build plan
    (``docs/WEBSERVICE_BUILD.md`` §6) it is the one deliberate exception, answered with the
    same error envelope but a **413** status instead of 422. Every other cap (factor/run
    counts, search budget, region rows, resolution, goals) is enforced inside the
    routers/``convert.py`` against ``doe_service.limits.DEFAULT_LIMITS`` directly, matching
    how the Milestone 4/5 goal/resolution caps already do it.
    """
    app = FastAPI(title="doe-service", version=__version__)
    app.state.limits = limits
    register_exception_handlers(app)
    app.include_router(designs.router)
    app.include_router(analysis.router)
    app.include_router(optimize.router)
    app.include_router(plot_data.router)
    app.add_middleware(BodySizeLimitMiddleware, limits=limits)

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "doe_version": version("doe")}

    return app
