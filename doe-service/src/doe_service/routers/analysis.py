"""Model fitting, ANOVA, prediction, and design diagnostics (Milestone 3).

Contracts: ``docs/WEBSERVICE_API.md`` "Analysis"; build steps:
``docs/WEBSERVICE_BUILD.md`` §3. Every endpoint that fits re-fits from ``{design,
response, model}`` internally — no fit handle crosses the wire.

Shared prologue for ``/fit``, ``/anova``, ``/predict``: :func:`design_from_document` ->
check ``response`` names a real run column (else 422 ``validation_error`` listing the
available columns, via :func:`_check_response_column`) -> resolve the model spec ->
``fit_ols`` inside :func:`~doe_service.convert.captured_warnings`, routed through
:func:`~doe_service.convert.call_library` so a library ``ValueError`` becomes 422
``infeasible``. ``/diagnostics`` and ``/coverage`` take no ``response`` and skip the fit
part of the prologue entirely -- they judge the design itself, before any experiment runs.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter

from doe import Design, Efficiency, FitResult, ValidationError
from doe import anova_records as _anova_records
from doe import condition_number as _condition_number
from doe import correlation_matrix as _correlation_matrix
from doe import discrepancy as _discrepancy
from doe import efficiency as _efficiency
from doe import fit_ols as _fit_ols
from doe import leverage as _leverage
from doe import maximin_distance as _maximin_distance
from doe import vif as _vif
from doe.analysis.fit import ModelSpec as _LibraryModelSpec
from doe.analysis.model import ModelMatrix
from doe.analysis.model import build_model_matrix as _build_model_matrix
from doe_service.convert import (
    ModelSpecLike,
    call_library,
    captured_warnings,
    design_from_document,
    jsonable,
    resolve_model,
)
from doe_service.convert import region_array as _region_array
from doe_service.schemas.analysis import (
    AnalysisFitRequest,
    AnovaRequest,
    CoverageRequest,
    DiagnosticsRequest,
    FitRequest,
    PredictRequest,
)
from doe_service.schemas.results import (
    AnovaResponse,
    CorrelationMatrixResult,
    CoverageResponse,
    DiagnosticsResponse,
    EfficiencyResult,
    FitResponse,
    PredictResponse,
)

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])

#: Default model for every endpoint that re-fits -- matches ``fit_ols``'s own default
#: (docs/WEBSERVICE_API.md "Model specification": "fit: linear").
_DEFAULT_FIT_MODEL = "linear"

#: ``/diagnostics``' own default (docs/WEBSERVICE_API.md "Model specification":
#: "diagnostics: {'order': 1, 'interactions': true}").
_DEFAULT_DIAGNOSTICS_MODEL: dict[str, object] = {"order": 1, "interactions": True}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _check_response_column(design: Design, response: str) -> None:
    """422 ``validation_error`` (not ``infeasible``) for a response-name typo.

    Checked *before* the library call so the mismatch gets the spec's
    ``validation_error`` code with the available columns listed in ``.errors``, rather
    than letting ``fit_ols`` raise its own identically-worded ``ValueError`` -- which
    :func:`call_library` would collapse into the generic ``infeasible`` code.
    """
    if response not in design.runs.columns:
        raise ValidationError(
            [
                f"no response column {response!r} on the design; "
                f"available columns: {list(design.runs.columns)}"
            ]
        )


def _resolved_fit(
    design: Design, response: str, model: ModelSpecLike | None, *, default: str
) -> FitResult:
    """Fit ``response`` on ``design`` under ``model`` (or ``default``).

    A *string* spec (``"linear"``/``"quadratic"``/``"scheffe-linear"``/
    ``"scheffe-quadratic"``) is passed straight through to ``fit_ols(..., model=...)``
    so its own all-mixture-design check runs for the Scheffé names; an
    ``{order, interactions}`` object is resolved to a plain ``order``/``interactions``
    call (an object carries no Scheffé name, so no such check applies).
    """
    spec = model if model is not None else default
    if isinstance(spec, str):
        # fit_ols validates the name itself (raising ValueError -- caught by
        # call_library -- for anything outside MODEL_SPECS); the cast only satisfies
        # its narrower Literal parameter type, it does not skip that runtime check.
        return _fit_ols(design, response, model=cast(_LibraryModelSpec, spec))
    order, interactions = resolve_model(spec, default=default)
    return _fit_ols(design, response, order=order, interactions=interactions)


def _fit(body: FitRequest, *, default: str = _DEFAULT_FIT_MODEL) -> tuple[FitResult, list[str]]:
    """Shared prologue for ``/fit``, ``/anova``, ``/predict``."""
    design = design_from_document(body.design.model_dump())
    _check_response_column(design, body.response)

    def run() -> FitResult:
        return _resolved_fit(design, body.response, body.model, default=default)

    with captured_warnings() as warns:
        result = call_library(run)
    return result, warns


def _points_frame(points: list[dict[str, Any]], factor_names: list[str]) -> pd.DataFrame:
    """Natural-unit ``points`` records -> a frame, naming every missing factor by record.

    ``FitResult.predict`` itself only reports missing factor columns in aggregate (across
    every point at once); the spec wants the offending record named too, so this checks
    each record before building the frame rather than delegating to the library's own
    (coarser) check.
    """
    problems = [
        f"point {i}: missing factor {name!r}"
        for i, record in enumerate(points)
        for name in factor_names
        if name not in record
    ]
    if problems:
        raise ValidationError(problems)
    if not points:
        return pd.DataFrame(columns=factor_names)
    return pd.DataFrame.from_records(points)[factor_names]


# --------------------------------------------------------------------------- #
# /fit
# --------------------------------------------------------------------------- #


@router.post("/fit")
def fit(body: AnalysisFitRequest) -> FitResponse:
    """Wraps ``doe.fit_ols`` → ``FitResult.to_dict``."""
    result, warns = _fit(body)
    payload = result.to_dict(confidence=body.confidence)
    return FitResponse.model_validate({**payload, "warnings": warns})


# --------------------------------------------------------------------------- #
# /anova
# --------------------------------------------------------------------------- #


@router.post("/anova")
def anova(body: AnovaRequest) -> AnovaResponse:
    """Wraps ``anova_records`` + ``lack_of_fit`` + ``press``/``predicted_r2``.

    ``lack_of_fit``'s "needs replicates" ``ValueError`` becomes ``lack_of_fit: null``
    plus a ``"no_pure_error"`` warning rather than an error response; ``press``/
    ``predicted_r2`` are null-guarded the same way for a saturated fit (no residual
    degrees of freedom to build a leave-one-out estimate from).
    """
    result, warns = _fit(body)
    rows = _anova_records(result)

    lack_of_fit: dict[str, Any] | None
    try:
        lack_of_fit = result.lack_of_fit().to_dict()
    except ValueError:
        lack_of_fit = None
        warns.append("no_pure_error")

    press_value: float | None
    predicted_r2_value: float | None
    if result.dof_resid <= 0:
        press_value = None
        predicted_r2_value = None
    else:
        with captured_warnings() as press_warns:
            press_value = cast(float | None, jsonable(result.press()))
            predicted_r2_value = cast(float | None, jsonable(result.predicted_r2()))
        warns.extend(press_warns)

    return AnovaResponse.model_validate(
        {
            "rows": rows,
            "lack_of_fit": lack_of_fit,
            "press": press_value,
            "predicted_r2": predicted_r2_value,
            "warnings": warns,
        }
    )


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


@router.post("/predict")
def predict(body: PredictRequest) -> PredictResponse:
    """Wraps ``FitResult.predict`` over natural-unit point records."""
    result, _warns = _fit(body)
    frame = _points_frame(body.points, result.factors.names)

    def run() -> list[float]:
        predicted = np.asarray(result.predict(frame), dtype=float)
        return cast(list[float], jsonable(predicted.tolist()))

    predictions = call_library(run)
    return PredictResponse(predictions=predictions)


# --------------------------------------------------------------------------- #
# /diagnostics
# --------------------------------------------------------------------------- #


@router.post("/diagnostics")
def diagnostics(body: DiagnosticsRequest) -> DiagnosticsResponse:
    """Wraps ``efficiency``/``vif``/``condition_number``/``correlation_matrix``/``leverage``.

    No ``response`` -- these judge the design against a model before any experiment runs.
    """
    design = design_from_document(body.design.model_dump())

    def run() -> tuple[ModelMatrix, Efficiency]:
        order, interactions = resolve_model(body.model, default=_DEFAULT_DIAGNOSTICS_MODEL)
        mm = _build_model_matrix(design, order=order, interactions=interactions)
        region = _region_array(body.region, n_factors=len(design.factors))
        eff = _efficiency(design, order=order, interactions=interactions, region=region)
        return mm, eff

    with captured_warnings():
        mm, eff = call_library(run)

    vif_series = _vif(mm.X, term_names=mm.term_names)
    vif_dict = {
        str(term): cast(float | None, jsonable(value)) for term, value in vif_series.items()
    }
    condition = cast(float | None, jsonable(_condition_number(mm.X)))
    corr = _correlation_matrix(mm.X, mm.term_names)
    lev = _leverage(mm.X)

    return DiagnosticsResponse(
        efficiency=EfficiencyResult(**eff.to_dict()),
        condition_number=condition,
        vif=vif_dict,
        correlation_matrix=CorrelationMatrixResult(
            labels=[str(label) for label in corr.index], matrix=corr.to_numpy().tolist()
        ),
        leverage=lev.tolist(),
    )


# --------------------------------------------------------------------------- #
# /coverage
# --------------------------------------------------------------------------- #


@router.post("/coverage")
def coverage(body: CoverageRequest) -> CoverageResponse:
    """Wraps ``doe.discrepancy`` + ``doe.maximin_distance``."""
    design = design_from_document(body.design.model_dump())

    def run() -> tuple[float, float]:
        return _discrepancy(design, method=body.method), _maximin_distance(design)

    with captured_warnings():
        disc, mm_dist = call_library(run)
    return CoverageResponse(discrepancy=disc, maximin_distance=mm_dist)
