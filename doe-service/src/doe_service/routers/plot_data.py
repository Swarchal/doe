"""Headless plot cores as JSON for frontend rendering (Milestone 5).

Contracts: ``docs/WEBSERVICE_API.md`` "Plot data"; build steps:
``docs/WEBSERVICE_BUILD.md`` §5. Thin wrappers over the *headless cores* in
``doe.plotting`` (``surface_grid``/``interaction_lines``/``ternary_grid``/
``alias_matrix``) -- no matplotlib is imported here; the cores return arrays, the router
JSON-encodes them. Pareto / main-effects / half-normal plots need no endpoint -- they
render directly from ``/v1/analysis/fit``'s ``terms``.

Two failure modes need router-level guards because ``call_library`` converts only
``ValueError`` -> 422 ``infeasible``: a categorical plot axis makes the cores raise
``TypeError`` and an unknown axis name makes them raise ``KeyError`` (both via
``plotting._require_continuous_axes``). Left uncaught, either would answer a well-formed
request with a 500, so :func:`_require_continuous_axes` checks them up front.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from doe import ContinuousFactor, FactorSet, ValidationError
from doe.plotting import alias_matrix as _alias_matrix
from doe.plotting import interaction_lines as _interaction_lines
from doe.plotting import surface_grid as _surface_grid
from doe.plotting import ternary_grid as _ternary_grid
from doe_service.convert import (
    call_library,
    design_from_document,
    jsonable,
    resolve_model,
)
from doe_service.errors import Infeasible
from doe_service.limits import DEFAULT_LIMITS, LimitExceeded
from doe_service.routers.analysis import _fit
from doe_service.schemas.plotting import (
    AliasRequest,
    InteractionsRequest,
    SurfaceRequest,
    TernaryRequest,
)
from doe_service.schemas.results import (
    AliasResponse,
    InteractionLine,
    InteractionsResponse,
    SurfaceResponse,
    TernaryResponse,
)

router = APIRouter(prefix="/v1/plot-data", tags=["plot-data"])

#: ``/surface``/``/interactions`` re-fit a *response surface*, so quadratic is the sensible
#: default; ``/ternary`` fits a Scheffé blending model (its default lives in ``_ternary``).
_DEFAULT_SURFACE_MODEL = "quadratic"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _check_resolution(resolution: int) -> None:
    """422 ``limit_exceeded`` for a ``resolution`` above the deployment cap.

    A grid is ``resolution^2`` (surface) or ``~resolution^2 / 2`` (ternary) predictions;
    the cap keeps a single request synchronous.
    """
    if resolution > DEFAULT_LIMITS.max_resolution:
        raise LimitExceeded(
            f"resolution {resolution} exceeds the cap of {DEFAULT_LIMITS.max_resolution}"
        )


def _require_continuous_axes(factors: FactorSet, axes: tuple[str, ...]) -> None:
    """Guard the plot axes before the core would raise a ``call_library``-invisible error.

    An unknown axis name is a request-shape problem -> 422 ``validation_error`` naming the
    available factors; a categorical axis has no coded sweep -> 422 ``infeasible``. Left to
    the library these surface as ``KeyError``/``TypeError``, which ``call_library`` (which
    converts only ``ValueError``) would let escape as a 500.
    """
    names = factors.names
    unknown = [name for name in axes if name not in names]
    if unknown:
        raise ValidationError(
            [f"unknown factor name(s) {unknown}; available factors: {names}"]
        )
    categorical = [name for name in axes if not isinstance(factors[name], ContinuousFactor)]
    if categorical:
        raise Infeasible(
            f"plot axes must be continuous; got non-continuous factor(s) {categorical} "
            "(a categorical factor has no coded [-1, +1] sweep)"
        )


# --------------------------------------------------------------------------- #
# /surface
# --------------------------------------------------------------------------- #


@router.post("/surface")
def surface(body: SurfaceRequest) -> SurfaceResponse:
    """Wraps ``doe.surface_grid`` (contour + 3-D surface meshes)."""
    _check_resolution(body.resolution)
    result, _ = _fit(body, default=_DEFAULT_SURFACE_MODEL)
    _require_continuous_axes(result.factors, (body.x, body.y))

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _surface_grid(
            result, body.x, body.y, fixed=body.fixed, resolution=body.resolution
        )

    nat_x, nat_y, z = call_library(run)
    return SurfaceResponse.model_validate(
        {"x": jsonable(nat_x), "y": jsonable(nat_y), "z": jsonable(z)}
    )


# --------------------------------------------------------------------------- #
# /interactions
# --------------------------------------------------------------------------- #


@router.post("/interactions")
def interactions(body: InteractionsRequest) -> InteractionsResponse:
    """Wraps ``doe.interaction_lines``."""
    _check_resolution(body.resolution)
    result, _ = _fit(body, default=_DEFAULT_SURFACE_MODEL)
    _require_continuous_axes(result.factors, (body.x, body.trace))

    def run() -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
        return _interaction_lines(
            result,
            body.x,
            body.trace,
            fixed=body.fixed,
            trace_levels=body.trace_levels,
            resolution=body.resolution,
        )

    nat_x, lines = call_library(run)
    return InteractionsResponse.model_validate(
        {
            "x": jsonable(nat_x),
            "lines": [
                InteractionLine(trace_value=value, z=jsonable(z))  # type: ignore[arg-type]
                for value, z in lines
            ],
        }
    )


# --------------------------------------------------------------------------- #
# /ternary
# --------------------------------------------------------------------------- #


@router.post("/ternary")
def ternary(body: TernaryRequest) -> TernaryResponse:
    """Wraps ``doe.ternary_grid`` (3-component Scheffé blending surface).

    A non-mixture or non-3-component fit makes the core raise ``ValueError``, which
    ``call_library`` maps to 422 ``infeasible`` -- no extra guard needed.
    """
    _check_resolution(body.resolution)
    result, _ = _fit(body, default="scheffe-quadratic")

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _ternary_grid(result, resolution=body.resolution)

    x, y, z, points = call_library(run)
    return TernaryResponse.model_validate(
        {"x": jsonable(x), "y": jsonable(y), "z": jsonable(z), "points": jsonable(points)}
    )


# --------------------------------------------------------------------------- #
# /alias
# --------------------------------------------------------------------------- #


@router.post("/alias")
def alias(body: AliasRequest) -> AliasResponse:
    """Wraps ``doe.alias_matrix`` (term correlation / alias structure).

    Judges the design itself -- no ``response`` and no fit; ``model`` picks which terms'
    aliasing is assessed, exactly as ``build_model_matrix`` does.
    """
    design = design_from_document(body.design.model_dump())
    order, inter = resolve_model(body.model, default="linear")

    def run() -> tuple[list[str], np.ndarray]:
        return _alias_matrix(design, order=order, interactions=inter, absolute=body.absolute)

    labels, matrix = call_library(run)
    return AliasResponse.model_validate({"labels": labels, "matrix": jsonable(matrix)})
