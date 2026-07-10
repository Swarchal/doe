"""Application factory.

Run locally with ``uvicorn --factory doe_service.main:create_app``.
"""

from importlib.metadata import version

from fastapi import FastAPI

from doe_service import __version__


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers mounted."""
    app = FastAPI(title="doe-service", version=__version__)

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "doe_version": version("doe")}

    return app
