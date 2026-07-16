"""Surface optimization and desirability (Milestone 4).

Contracts: ``docs/WEBSERVICE_API.md`` "Optimization"; build steps:
``docs/WEBSERVICE_BUILD.md`` §4. ``/stationary-point``/``/optimum`` reuse the analysis
router's shared fit prologue (``routers.analysis._fit``/``_check_response_column``/
``_resolved_fit``) rather than reinventing it -- this milestone fits exactly the way
Milestone 3 does, then feeds the result into surface optimization instead of returning
it directly. Both require a quadratic model (the library's ``_quadratic_form`` needs
curvature to locate); a linear model is rejected here as 422 ``infeasible`` before any
fit is attempted, so the wasted fit never runs.

All three endpoints additionally require all-continuous factors: they search the coded
box, and the library signals a mixture/categorical fit with a bare ``TypeError`` that
would otherwise escape ``call_library`` as a 500 (see ``_require_continuous_factors``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from doe import (
    CategoricalFactor,
    CategoricalOptimum,
    ContinuousFactor,
    DesirabilityResult,
    FactorSet,
    FitResult,
    Optimum,
    ResponseGoal,
    ValidationError,
)
from doe import categorical_optimum as _categorical_optimum
from doe import desirability as _desirability
from doe import optimum as _optimum
from doe import stationary_point as _stationary_point
from doe_service.convert import (
    ModelSpecLike,
    call_library,
    captured_warnings,
    design_from_document,
    resolve_model,
)
from doe_service.deps import app_limits
from doe_service.errors import Infeasible
from doe_service.limits import LimitExceeded, Limits
from doe_service.routers.analysis import _check_response_column, _fit, _resolved_fit
from doe_service.schemas.common import Bounds
from doe_service.schemas.optimization import (
    DesirabilityRequest,
    GoalSchema,
    OptimumRequest,
    StationaryPointRequest,
)
from doe_service.schemas.results import (
    CategoricalOptimumResponse,
    DesirabilityResponse,
    OptimumResponse,
    StationaryPointResponse,
)

router = APIRouter(prefix="/v1/optimize", tags=["optimize"])

#: Both surface-optimization endpoints need curvature (a non-degenerate quadratic form)
#: to locate, so quadratic is the sensible default when ``model`` is omitted -- the
#: analysis router's own "linear" default would always fail the check below.
_DEFAULT_OPTIMIZE_MODEL = "quadratic"

#: ``/desirability``'s per-goal default -- matches ``fit_ols``'s own default
#: (``docs/WEBSERVICE_API.md`` "Model specification": "fit: linear"); unlike
#: stationary-point/optimum, a desirability goal does not require curvature (see the
#: worked example's ``"cost"`` goal, fitted ``model: "linear"``).
_DEFAULT_GOAL_MODEL = "linear"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _require_quadratic(model: ModelSpecLike | None) -> None:
    """422 ``infeasible`` for anything but a quadratic model (order=2).

    ``stationary_point``/``optimum`` read the fit as ``y-hat = b0 + x^T b + x^T B x``
    (the library's ``_quadratic_form``); a linear fit has no ``B`` and so no stationary
    point/curvature to search, checked here before the (otherwise wasted) fit runs.
    """

    def resolve() -> tuple[int, bool]:
        return resolve_model(model, default=_DEFAULT_OPTIMIZE_MODEL)

    order, _ = call_library(resolve)
    if order != 2:
        raise Infeasible(
            "stationary-point/optimum require a quadratic model (order=2); "
            f"got order={order} from model={model!r}"
        )


def _require_continuous_factors(factors: FactorSet) -> None:
    """422 ``infeasible`` for surface optimization over non-continuous factors.

    All three endpoints search the coded box, which a mixture (simplex) or categorical
    factor has no place in. The library says so by raising a bare ``TypeError`` from
    ``_quadratic_form``, which ``call_library`` (``ValueError`` only, deliberately) lets
    escape as a 500 -- but such a request is a domain infeasibility, not a service bug,
    so it must land as 422 like every other one.
    """
    non_continuous = [
        name for name in factors.names if not isinstance(factors[name], ContinuousFactor)
    ]
    if non_continuous:
        raise Infeasible(
            "surface optimization requires all-continuous factors (the coded box); got "
            f"non-continuous factor(s) {non_continuous} -- for mixture fits read the "
            "optimum off /v1/plot-data/ternary instead"
        )


def _require_optimizable_factors(factors: FactorSet) -> None:
    """422 ``infeasible`` for a factor ``categorical_optimum`` cannot handle.

    It optimizes continuous factors within each categorical-level combination; a mixture
    (simplex) factor belongs to neither kind and makes the library raise a bare
    ``TypeError`` that ``call_library`` (``ValueError`` only) would let escape as a 500.
    """
    non_optimizable = [
        name
        for name in factors.names
        if not isinstance(factors[name], (ContinuousFactor, CategoricalFactor))
    ]
    if non_optimizable:
        raise Infeasible(
            "categorical_optimum handles continuous and categorical factors only; got "
            f"factor(s) {non_optimizable} of another kind (e.g. mixture) -- for a mixture "
            "fit read the optimum off /v1/plot-data/ternary instead"
        )


def _fit_quadratic(
    body: StationaryPointRequest | OptimumRequest, limits: Limits
) -> tuple[FitResult, list[str]]:
    """Shared prologue: reject a non-quadratic model, then the M3 fit prologue."""
    _require_quadratic(body.model)
    result, warns = _fit(body, default=_DEFAULT_OPTIMIZE_MODEL, limits=limits)
    _require_continuous_factors(result.factors)
    return result, warns


def _validate_bounds(bounds: Bounds | None, factor_names: list[str]) -> Bounds | None:
    """Unknown factor name in a natural-units ``bounds`` mapping -> 422 ``validation_error``.

    Checked before the library call, whose own equivalent check raises a bare
    ``ValueError`` that :func:`~doe_service.convert.call_library` would collapse into the
    generic ``infeasible`` code -- bounds naming an unknown factor is a request-shape
    problem, not a domain infeasibility.
    """
    if isinstance(bounds, dict):
        unknown = sorted(set(bounds) - set(factor_names))
        if unknown:
            raise ValidationError(
                [f"unknown factor name(s) in bounds: {unknown}; valid names are {factor_names}"]
            )
    return bounds


# --------------------------------------------------------------------------- #
# /stationary-point
# --------------------------------------------------------------------------- #


@router.post("/stationary-point")
def stationary_point(
    body: StationaryPointRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> StationaryPointResponse:
    """Wraps ``doe.stationary_point`` (canonical analysis of the fitted quadratic)."""
    result, warns = _fit_quadratic(body, limits)
    sp = call_library(_stationary_point, result)
    payload = sp.to_dict()
    return StationaryPointResponse.model_validate({**payload, "warnings": warns})


# --------------------------------------------------------------------------- #
# /optimum
# --------------------------------------------------------------------------- #


@router.post("/optimum")
def optimum(
    body: OptimumRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> OptimumResponse:
    """Wraps ``doe.optimum`` (constrained multistart search over the coded box)."""
    result, warns = _fit_quadratic(body, limits)
    bounds = _validate_bounds(body.bounds, result.factors.names)
    bounds_arg: Bounds = bounds if bounds is not None else (-1.0, 1.0)

    def run() -> Optimum:
        return _optimum(result, maximize=body.maximize, bounds=bounds_arg)

    opt = call_library(run)
    payload = opt.to_dict()
    return OptimumResponse.model_validate({**payload, "warnings": warns})


# --------------------------------------------------------------------------- #
# /categorical-optimum
# --------------------------------------------------------------------------- #


@router.post("/categorical-optimum")
def categorical_optimum(
    body: OptimumRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> CategoricalOptimumResponse:
    """Wraps ``doe.categorical_optimum`` -- the mixed continuous/categorical optimum.

    ``/optimum`` searches a continuous coded box and so rejects a categorical factor
    (422). This endpoint instead enumerates every combination of the categorical factors'
    levels, optimizes the continuous factors exactly within each, and returns the best --
    naming the winning ``levels``. An all-continuous fit is accepted too (it simply
    delegates to ``optimum`` with empty ``levels``), so a caller need not know in advance
    whether the design has a categorical factor. A mixture (simplex) fit has no coded box
    either and raises ``TypeError`` -> 422 ``infeasible`` (via ``call_library``).

    Requires a quadratic model, exactly like ``/optimum``.
    """
    _require_quadratic(body.model)
    result, warns = _fit(body, default=_DEFAULT_OPTIMIZE_MODEL, limits=limits)
    _require_optimizable_factors(result.factors)
    bounds = _validate_bounds(body.bounds, result.factors.names)
    bounds_arg: Bounds = bounds if bounds is not None else (-1.0, 1.0)

    def run() -> CategoricalOptimum:
        return _categorical_optimum(result, maximize=body.maximize, bounds=bounds_arg)

    opt = call_library(run)
    payload = opt.to_dict()
    return CategoricalOptimumResponse.model_validate({**payload, "warnings": warns})


# --------------------------------------------------------------------------- #
# /desirability
# --------------------------------------------------------------------------- #


@router.post("/desirability")
def desirability(
    body: DesirabilityRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesirabilityResponse:
    """Wraps ``doe.desirability`` over per-goal re-fits (``doe.ResponseGoal``).

    Each goal re-fits its own ``response``/``model`` on the shared design -- the
    library's bracket ``ValueError``s (e.g. a ``target`` outside ``(low, high)``) become
    422 ``infeasible``. Goal count is capped at ``limits.max_goals``.
    """
    if len(body.goals) > limits.max_goals:
        raise LimitExceeded(
            f"too many desirability goals: {len(body.goals)} exceeds the cap of "
            f"{limits.max_goals}"
        )

    design = design_from_document(body.design.model_dump(), limits=limits)
    _require_continuous_factors(design.factors)
    bounds = _validate_bounds(body.bounds, design.factors.names)
    bounds_arg: Bounds = bounds if bounds is not None else (-1.0, 1.0)

    warns: list[str] = []
    goals: list[ResponseGoal] = []
    with captured_warnings() as fit_warns:
        for entry in body.goals:
            _check_response_column(design, entry.response)
            fit = call_library(
                _resolved_fit, design, entry.response, entry.model, default=_DEFAULT_GOAL_MODEL
            )

            def make_goal(entry: GoalSchema = entry, fit: FitResult = fit) -> ResponseGoal:
                return ResponseGoal(
                    result=fit,
                    goal=entry.goal,
                    low=entry.low,
                    high=entry.high,
                    target=entry.target,
                    weight=entry.weight,
                )

            goals.append(call_library(make_goal))
    warns.extend(fit_warns)

    def run() -> DesirabilityResult:
        return _desirability(goals, bounds=bounds_arg)

    result = call_library(run)
    payload = result.to_dict()
    return DesirabilityResponse.model_validate({**payload, "warnings": warns})
