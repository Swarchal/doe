"""Request models for the analysis endpoints (Milestone 3).

Contracts: ``docs/WEBSERVICE_API.md`` "Analysis"; build steps:
``docs/WEBSERVICE_BUILD.md`` §3. Every endpoint that fits takes ``{design, response,
model}`` and re-fits internally -- no fit handle crosses the wire.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


#: Caps on the free-form report ``sections`` (the body-size cap bounds them too, but a
#: section count/row count cap keeps a pathological request from rendering for ages).
MAX_REPORT_SECTIONS = 20
MAX_REPORT_SECTION_ROWS = 1000


class ReportResponseSpec(BaseModel):
    """One response to fit and report: ``{response, model?}`` (model defaults as ``/fit``)."""

    response: str
    model: ModelSpec | None = None


class ReportSection(BaseModel):
    """A caller-supplied extra section: a heading plus optional prose and/or a table.

    ``rows`` are table records (``[{column: value}, ...]``), rendered in order -- typically
    results the client already computed and wants on the record (best settings, a
    desirability balance, a per-level comparison).
    """

    heading: str = Field(min_length=1, max_length=200)
    text: str | None = Field(default=None, max_length=20_000)
    rows: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_REPORT_SECTION_ROWS)


class ReportRequest(DesignRequest):
    """``POST /v1/analysis/report`` -- render fits of one or more responses as HTML."""

    responses: list[ReportResponseSpec] = Field(min_length=1)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=20_000)
    sections: list[ReportSection] = Field(default_factory=list, max_length=MAX_REPORT_SECTIONS)
    figures: bool = True
    image_format: Literal["png", "svg"] = "png"
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
