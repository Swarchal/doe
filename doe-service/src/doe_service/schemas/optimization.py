"""Request models for the optimization endpoints (Milestone 4).

Contracts: ``docs/WEBSERVICE_API.md`` "Optimization"; build steps:
``docs/WEBSERVICE_BUILD.md`` §4. ``StationaryPointRequest``/``OptimumRequest`` reuse the
shared ``{design, response, model}`` fit request from the analysis router
(``schemas/analysis.py``) rather than redefining it -- Milestone 4 fits exactly the way
Milestone 3 does, then feeds the result into surface optimization instead of returning
it directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from doe_service.schemas.analysis import FitRequest
from doe_service.schemas.common import Bounds, ModelSpec
from doe_service.schemas.design import DesignRequest


class StationaryPointRequest(FitRequest):
    """``POST /v1/optimize/stationary-point`` takes no parameters beyond the shared fit."""


class OptimumRequest(FitRequest):
    """``POST /v1/optimize/optimum`` -- adds the search direction and the search box."""

    maximize: bool = True
    bounds: Bounds | None = None


class GoalSchema(BaseModel):
    """One entry of ``POST /v1/optimize/desirability``'s ``goals`` array.

    Mirrors ``doe.ResponseGoal`` plus the ``response``/``model`` needed to re-fit it
    server-side -- the library's own ``ResponseGoal.to_dict`` deliberately omits the fit
    for the same reason (it is *derived*, re-fit on load, not data worth persisting).
    """

    response: str
    model: ModelSpec | None = None
    goal: Literal["max", "min", "target"]
    low: float
    high: float
    target: float | None = None
    weight: float = 1.0


class DesirabilityRequest(DesignRequest):
    """``POST /v1/optimize/desirability`` -- one design carrying every response column,
    plus a goal per response and the shared search box."""

    goals: list[GoalSchema]
    bounds: Bounds | None = None
