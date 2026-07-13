"""Typed response models for the analysis, optimize, and plot-data endpoints.

These land per milestone (``docs/WEBSERVICE_BUILD.md`` §3–§5) and mirror the library
``to_dict`` shapes added in Milestone 0 — the shapes live and are tested in ``doe``;
these models only *declare* them for OpenAPI. Milestone 3 (analysis router) adds
``FitResponse``, ``AnovaResponse``, ``PredictResponse``, ``DiagnosticsResponse``,
``CoverageResponse``. Planned for later milestones: ``StationaryPointResponse``,
``OptimumResponse``, ``DesirabilityResponse``, ``SurfaceResponse``,
``InteractionsResponse``, ``TernaryResponse``, ``AliasResponse``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

#: Milestone 6 OpenAPI polish (``docs/WEBSERVICE_BUILD.md`` §6): every response model
#: below whose value is small enough to embed cheaply gets a real ``examples`` entry --
#: the exact JSON captured in ``doe-service/tests/contract/pairs/`` (a real, seeded fit
#: of a 2-factor central-composite design), not a hand-invented placeholder. Response
#: models that would have to embed a full ``DesignDocument`` (``DesignResponse``,
#: ``OptimalDesignResponse``) are left without one -- not "cheap".

# --------------------------------------------------------------------------- #
# /v1/analysis/fit
# --------------------------------------------------------------------------- #


class ResolvedModel(BaseModel):
    """The resolved ``(order, interactions)`` model spec, echoed on fit responses."""

    order: int
    interactions: bool


class TermResult(BaseModel):
    """One row of ``FitResult.to_dict()["terms"]``.

    ``effect``/``std_error``/``t``/``p``/``ci_low``/``ci_high`` are ``null`` for a
    saturated model's undefined inference columns, and ``effect`` alone is ``null`` for
    every term of a Scheffé (mixture) fit -- the ±1 swing is meaningless on proportions.
    """

    term: str
    coefficient: float
    effect: float | None
    std_error: float | None
    t: float | None
    p: float | None
    ci_low: float | None
    ci_high: float | None


class FitResponse(BaseModel):
    """``POST /v1/analysis/fit`` response -- ``FitResult.to_dict()`` plus ``warnings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "terms": [
                        {
                            "term": "Intercept",
                            "coefficient": 70.10705699780807,
                            "effect": 70.10705699780807,
                            "std_error": 0.09453330855157353,
                            "t": 741.6122218927787,
                            "p": 2.140522787971692e-18,
                            "ci_low": 69.88352124382372,
                            "ci_high": 70.33059275179242,
                        },
                        {
                            "term": "temp",
                            "coefficient": 2.133386638852627,
                            "effect": 4.266773277705254,
                            "std_error": 0.09294440028775322,
                            "t": 22.953363863209862,
                            "p": 7.551182348598893e-08,
                            "ci_low": 1.9136080558824584,
                            "ci_high": 2.3531652218227954,
                        },
                    ],
                    "r_squared": 0.9965954680087256,
                    "adjusted_r2": 0.9941636594435296,
                    "dof_resid": 7,
                    "mse": 0.051831969269100595,
                    "fitted": [62.441177194293154, 69.44067289708941],
                    "residuals": [-0.17911393684052257, 0.13149848797206687],
                    "model": {"order": 2, "interactions": True},
                    "warnings": [],
                }
            ]
        }
    )

    terms: list[TermResult]
    r_squared: float | None
    adjusted_r2: float | None
    dof_resid: int
    mse: float | None
    fitted: list[float]
    residuals: list[float]
    model: ResolvedModel
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# /v1/analysis/anova
# --------------------------------------------------------------------------- #


class AnovaRow(BaseModel):
    """One row of ``anova_records`` -- a model term, or the ``Residual``/``Total`` rows."""

    term: str
    ss: float
    df: float
    ms: float | None
    f: float | None
    p: float | None


class LackOfFitResult(BaseModel):
    """``LackOfFit.to_dict()``."""

    ss_lof: float
    df_lof: int
    ss_pe: float
    df_pe: int
    f: float | None
    p: float | None


class AnovaResponse(BaseModel):
    """``POST /v1/analysis/anova`` response.

    ``lack_of_fit`` is ``null`` (plus a ``"no_pure_error"`` warning) when the design has
    no replicated factor setting; ``press``/``predicted_r2`` are ``null`` for a saturated
    fit (no leave-one-out residual is defined without residual degrees of freedom).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "rows": [
                        {
                            "term": "temp", "ss": 27.308031305009465, "df": 1.0,
                            "ms": 27.308031305009465, "f": 526.8569126369086,
                            "p": 7.551182348598881e-08,
                        },
                        {
                            "term": "Residual", "ss": 0.36282378488370415, "df": 7.0,
                            "ms": 0.051831969269100595, "f": None, "p": None,
                        },
                        {
                            "term": "Total", "ss": 106.57082553889848, "df": 12.0,
                            "ms": None, "f": None, "p": None,
                        },
                    ],
                    "lack_of_fit": {
                        "ss_lof": 0.27583370370697446, "df_lof": 3,
                        "ss_pe": 0.08699008117672971, "df_pe": 4,
                        "f": 4.227818466591819, "p": 0.0987389979990755,
                    },
                    "press": 2.1289671696948753,
                    "predicted_r2": 0.9800229832233232,
                    "warnings": [],
                }
            ]
        }
    )

    rows: list[AnovaRow]
    lack_of_fit: LackOfFitResult | None
    press: float | None
    predicted_r2: float | None
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# /v1/analysis/predict
# --------------------------------------------------------------------------- #


class PredictResponse(BaseModel):
    """``POST /v1/analysis/predict`` response."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"predictions": [70.06864843973118, 63.038460205546464]}]}
    )

    predictions: list[float]


# --------------------------------------------------------------------------- #
# /v1/analysis/diagnostics
# --------------------------------------------------------------------------- #


class EfficiencyResult(BaseModel):
    """``Efficiency.to_dict()``."""

    d: float
    a: float
    g: float
    i: float


class CorrelationMatrixResult(BaseModel):
    """The alias/correlation matrix, labelled by (constant-dropped) model term."""

    labels: list[str]
    matrix: list[list[float]]


class DiagnosticsResponse(BaseModel):
    """``POST /v1/analysis/diagnostics`` response -- no ``response``, judges the design.

    ``condition_number``/``vif`` entries are ``null`` for a singular design / a fully
    aliased term respectively (both are ``inf`` in the library before JSON sanitizing).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "efficiency": {
                        "d": 0.5059797181517013, "a": 0.4660194174757281,
                        "g": 0.4660194174757281, "i": 0.5208333333333334,
                    },
                    "condition_number": 1.8027756377319946,
                    "vif": {"temp": 1.0, "time": 1.0, "temp:time": 1.0},
                    "correlation_matrix": {
                        "labels": ["temp", "time", "temp:time"],
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    "leverage": [0.6602564102564108, 0.24358974358974358],
                }
            ]
        }
    )

    efficiency: EfficiencyResult
    condition_number: float | None
    vif: dict[str, float | None]
    correlation_matrix: CorrelationMatrixResult
    leverage: list[float]


# --------------------------------------------------------------------------- #
# /v1/analysis/coverage
# --------------------------------------------------------------------------- #


class CoverageResponse(BaseModel):
    """``POST /v1/analysis/coverage`` response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"discrepancy": 0.04565253122945401, "maximin_distance": 0.0}]
        }
    )

    discrepancy: float
    maximin_distance: float


# --------------------------------------------------------------------------- #
# /v1/optimize/stationary-point
# --------------------------------------------------------------------------- #


class StationaryPointResponse(BaseModel):
    """``POST /v1/optimize/stationary-point`` response -- ``StationaryPoint.to_dict()``
    plus ``warnings``.

    Includes ``eigenvectors`` alongside ``eigenvalues`` (the spec's worked example was
    abridged -- see ``docs/WEBSERVICE_BUILD.md`` "Resolved decisions") and
    ``response_name`` (the fit's response column, echoed so a labelled response reports
    labelled).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "kind": "maximum",
                    "natural": {"temp": 62.43573493079263, "time": 7.245169319200981},
                    "coded": [0.4145244976930875, 0.44903386384019617],
                    "response": 70.86320899778093,
                    "eigenvalues": [-3.3961470300336924, -1.8377046290019483],
                    "eigenvectors": [
                        [-0.9242522041606367, -0.38178248140034543],
                        [0.38178248140034543, -0.9242522041606367],
                    ],
                    "response_name": "yield",
                    "warnings": [],
                }
            ]
        }
    )

    kind: Literal["maximum", "minimum", "saddle"]
    natural: dict[str, float]
    coded: list[float]
    response: float
    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    response_name: str | None
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# /v1/optimize/optimum
# --------------------------------------------------------------------------- #


class OptimumResponse(BaseModel):
    """``POST /v1/optimize/optimum`` response -- ``Optimum.to_dict()`` plus ``warnings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "natural": {"temp": 62.435734951761404, "time": 7.245169322626485},
                    "coded": [0.4145244983920467, 0.44903386452529703],
                    "response": 70.86320899778094,
                    "maximize": True,
                    "at_bound": False,
                    "response_name": "yield",
                    "warnings": [],
                }
            ]
        }
    )

    natural: dict[str, float]
    coded: list[float]
    response: float
    maximize: bool
    at_bound: bool
    response_name: str | None
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# /v1/optimize/desirability
# --------------------------------------------------------------------------- #


class DesirabilityResponse(BaseModel):
    """``POST /v1/optimize/desirability`` response -- ``DesirabilityResult.to_dict()``
    plus ``warnings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "natural": {"temp": 49.1716649105463, "time": 4.585832455284189},
                    "coded": [-0.02761116964845672, -0.08283350894316223],
                    "overall": 0.5120549818503818,
                    "responses": {
                        "yield_pct": 69.34893318762082,
                        "impurity_pct": 1.922917437992454,
                        "cost": 9.9999999999989,
                    },
                    "individual": {
                        "yield_pct": 0.3116311062540272,
                        "impurity_pct": 0.4308330248030184,
                        "cost": 0.9999999999994502,
                    },
                    "warnings": [],
                }
            ]
        }
    )

    natural: dict[str, float]
    coded: list[float]
    overall: float
    responses: dict[str, float]
    individual: dict[str, float]
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# /v1/plot-data/surface
# --------------------------------------------------------------------------- #


class SurfaceResponse(BaseModel):
    """``POST /v1/plot-data/surface`` -- natural-unit ``(X, Y, Z)`` meshes.

    Each is a ``(resolution, resolution)`` nested list; ``surface_grid``'s three arrays
    routed through ``jsonable``.
    """

    x: list[list[float]]
    y: list[list[float]]
    z: list[list[float | None]]


# --------------------------------------------------------------------------- #
# /v1/plot-data/interactions
# --------------------------------------------------------------------------- #


class InteractionLine(BaseModel):
    """One ``trace``-level line: its natural value and the fitted ``z`` sweep."""

    trace_value: float
    z: list[float | None]


class InteractionsResponse(BaseModel):
    """``POST /v1/plot-data/interactions`` -- the ``x`` sweep and a line per trace level."""

    x: list[float]
    lines: list[InteractionLine]


# --------------------------------------------------------------------------- #
# /v1/plot-data/ternary
# --------------------------------------------------------------------------- #


class TernaryResponse(BaseModel):
    """``POST /v1/plot-data/ternary`` -- flat Cartesian ``x``/``y``, fitted ``z``, and the
    ``(n, 3)`` barycentric ``points`` they came from."""

    x: list[float]
    y: list[float]
    z: list[float | None]
    points: list[list[float]]


# --------------------------------------------------------------------------- #
# /v1/plot-data/alias
# --------------------------------------------------------------------------- #


class AliasResponse(BaseModel):
    """``POST /v1/plot-data/alias`` -- model-term labels and their correlation matrix."""

    labels: list[str]
    matrix: list[list[float | None]]
