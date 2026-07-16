"""Application factory.

Run locally with ``uv run doe-web`` (see :func:`main`), or directly with
``uvicorn --factory doe_web.main:create_app``.

One process serves both halves: the full doe-service API is mounted under ``/api``
(so ``POST /api/v1/designs/central-composite`` etc.), and the static single-page
front-end is served from ``/``. Same origin, so no CORS configuration is needed --
the doe-service app is mounted unmodified, limits, error envelopes and all.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from doe_service.limits import DEFAULT_LIMITS
from doe_service.main import create_app as create_api_app
from doe_web import __version__

STATIC_DIR = Path(__file__).parent / "static"

#: Above this many runs a D-optimal / augment search is heavy enough that parallelising its
#: restarts across cores wins back more than the worker start-up + pickling overhead costs
#: (below it, the single-process path is faster). doe-web is a single-user prototype, so the
#: search may use all cores (``-1``); the library still caps workers at the restart count. The
#: plain ``doe-service`` leaves this disabled -- only doe-web opts in here.
OPTIMAL_PARALLEL_MIN_RUNS = 24
OPTIMAL_PARALLEL_MAX_WORKERS = -1

WEB_LIMITS = replace(
    DEFAULT_LIMITS,
    optimal_parallel_min_runs=OPTIMAL_PARALLEL_MIN_RUNS,
    optimal_parallel_max_workers=OPTIMAL_PARALLEL_MAX_WORKERS,
)


def create_app() -> FastAPI:
    """Build the combined app: doe-service under ``/api``, the SPA at ``/``.

    The mounted API is given :data:`WEB_LIMITS` -- identical to the service defaults except
    that it enables auto-parallel optimal-design search for large run counts (see
    :meth:`doe_service.limits.Limits.optimal_n_jobs`), so big D-optimal/augment plans built
    from the browser use multiple cores.
    """
    app = FastAPI(title="doe-web", version=__version__)
    app.mount("/api", create_api_app(limits=WEB_LIMITS))
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    """Console entry point: serve the combined app with uvicorn.

    Installed as the ``doe-web`` script, so ``uv run doe-web`` launches the
    server (``--host``/``--port``/``--reload`` are forwarded to uvicorn).
    """
    parser = argparse.ArgumentParser(prog="doe-web", description="Serve the doe-web app.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="restart on source changes")
    args = parser.parse_args()
    uvicorn.run(
        "doe_web.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
