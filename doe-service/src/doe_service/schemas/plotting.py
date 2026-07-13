"""Request models for the plot-data endpoints (Milestone 5).

Contracts: ``docs/WEBSERVICE_API.md`` "Plot data"; build steps:
``docs/WEBSERVICE_BUILD.md`` §5. The surface/interactions/ternary endpoints re-fit from
``{design, response, model}`` (they draw a *fitted* surface), so they extend the shared
``FitRequest``; ``/alias`` judges the design itself and takes no ``response``.
"""

from __future__ import annotations

from typing import Any

from doe_service.schemas.analysis import FitRequest
from doe_service.schemas.common import ModelSpec
from doe_service.schemas.design import DesignRequest


class SurfaceRequest(FitRequest):
    """``POST /v1/plot-data/surface`` -- two continuous axes over the coded box."""

    x: str
    y: str
    fixed: dict[str, Any] | None = None
    resolution: int = 25


class InteractionsRequest(FitRequest):
    """``POST /v1/plot-data/interactions`` -- ``x`` swept, a line per ``trace`` level."""

    x: str
    trace: str
    fixed: dict[str, Any] | None = None
    trace_levels: list[float] | None = None
    resolution: int = 25


class TernaryRequest(FitRequest):
    """``POST /v1/plot-data/ternary`` -- a 3-component Scheffé blending surface."""

    resolution: int = 100


class AliasRequest(DesignRequest):
    """``POST /v1/plot-data/alias`` -- no ``response``; term-to-term alias structure."""

    model: ModelSpec | None = None
    absolute: bool = False
