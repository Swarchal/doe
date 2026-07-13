"""Application factory.

Run locally with ``uvicorn --factory doe_service.main:create_app``.
"""

from collections.abc import Awaitable, Callable
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from doe_service import __version__
from doe_service.errors import error_envelope, register_exception_handlers
from doe_service.limits import DEFAULT_LIMITS, Limits
from doe_service.routers import analysis, designs, optimize, plot_data

Endpoint = Callable[[Request], Awaitable[Response]]


def create_app(*, limits: Limits = DEFAULT_LIMITS) -> FastAPI:
    """Build the FastAPI application with all routers mounted.

    ``limits`` is stashed on ``app.state`` (``docs/WEBSERVICE_API.md`` "Limits": "All
    configurable at deployment") and read by the body-size middleware below, the one
    check that has to run ahead of routing -- a request whose ``Content-Length`` exceeds
    ``limits.max_body_bytes`` never reaches a router, so it cannot go through the usual
    ``LimitExceeded`` -> 422 path; per the build plan (``docs/WEBSERVICE_BUILD.md`` §6)
    it is the one deliberate exception, answered with the same error envelope but a
    **413** status instead of 422. Every other cap (factor/run counts, search budget,
    region rows, resolution, goals) is enforced inside the routers/``convert.py`` against
    ``doe_service.limits.DEFAULT_LIMITS`` directly, matching how the Milestone 4/5
    goal/resolution caps already do it.
    """
    app = FastAPI(title="doe-service", version=__version__)
    app.state.limits = limits
    register_exception_handlers(app)
    app.include_router(designs.router)
    app.include_router(analysis.router)
    app.include_router(optimize.router)
    app.include_router(plot_data.router)

    @app.middleware("http")
    async def enforce_body_size(request: Request, call_next: Endpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > request.app.state.limits.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content=error_envelope(
                        "limit_exceeded",
                        f"request body of {size} bytes exceeds the cap of "
                        f"{request.app.state.limits.max_body_bytes} bytes",
                    ),
                )
        return await call_next(request)

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "doe_version": version("doe")}

    return app
