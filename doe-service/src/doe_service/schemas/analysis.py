"""Request models for the analysis endpoints (Milestone 3).

Contracts: ``docs/WEBSERVICE_API.md`` "Analysis"; build steps:
``docs/WEBSERVICE_BUILD.md`` §3. Every endpoint that fits takes ``{design, response,
model}`` and re-fits internally -- no fit handle crosses the wire.
"""

from __future__ import annotations

from typing import Any, Literal

from doe_service.schemas.common import ModelSpec
from doe_service.schemas.design import DesignRequest


class FitRequest(DesignRequest):
    """Shared base for every endpoint that re-fits: ``{design, response, model}``."""

    response: str
    model: ModelSpec | None = None


class AnalysisFitRequest(FitRequest):
    """``POST /v1/analysis/fit`` -- adds the confidence level for the CI columns."""

    confidence: float = 0.95


class FitGlsRequest(FitRequest):
    """``POST /v1/analysis/fit-gls`` -- the split-plot GLS fit.

    Same ``{design, response, model}`` shape as ``/fit``, but the posted design must carry
    ``whole_plots`` (i.e. it came from ``split_plot`` or was serialized with a whole-plot
    structure); ``fit_gls`` estimates the two variance components by REML and returns the
    whole-plot-aware standard errors OLS understates.
    """

    confidence: float = 0.95


class AnovaRequest(FitRequest):
    """``POST /v1/analysis/anova`` takes no parameters beyond the shared fit."""


class PredictRequest(FitRequest):
    """``POST /v1/analysis/predict`` -- natural-unit point records to predict at."""

    points: list[dict[str, Any]]


class DiagnosticsRequest(DesignRequest):
    """``POST /v1/analysis/diagnostics`` -- no ``response``; judges the design itself."""

    model: ModelSpec | None = None
    region: list[list[float]] | None = None


class CoverageRequest(DesignRequest):
    """``POST /v1/analysis/coverage`` parameters."""

    method: Literal["CD", "WD", "MD", "L2-star"] = "CD"
