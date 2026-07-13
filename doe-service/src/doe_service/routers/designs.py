"""Design generation, operations, and candidate sets (Milestone 2).

Contracts: ``docs/WEBSERVICE_API.md`` "Design generation" / "Design operations";
build steps: ``docs/WEBSERVICE_BUILD.md`` §2.

Pattern: request model = ``factors: list[FactorSchema]`` + the library keywords named in
the API doc's table; body = build the factor list -> :func:`call_library` around the
generator -> :class:`DesignResponse`. Every route stays one call thin; the special cases
(``/optimal``, ``/augment``, ``/candidates``, ``/validate``) are noted inline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
from fastapi import APIRouter

from doe import (
    Design,
    Factor,
    FactorSet,
    OptimalDesign,
    ValidationError,
)
from doe import augment as _augment
from doe import box_behnken as _box_behnken
from doe import candidate_grid as _candidate_grid
from doe import central_composite as _central_composite
from doe import coordinate_exchange as _coordinate_exchange
from doe import definitive_screening as _definitive_screening
from doe import extreme_vertices as _extreme_vertices
from doe import fractional_factorial as _fractional_factorial
from doe import full_factorial as _full_factorial
from doe import halton as _halton
from doe import latin_hypercube as _latin_hypercube
from doe import mixture_candidates as _mixture_candidates
from doe import plackett_burman as _plackett_burman
from doe import simplex_centroid as _simplex_centroid
from doe import simplex_lattice as _simplex_lattice
from doe import sobol as _sobol
from doe import validate_design_dict as _validate_design_dict
from doe_service.convert import (
    call_library,
    captured_warnings,
    check_factor_count,
    check_run_count,
    check_search_budget,
    design_from_document,
    jsonable,
)
from doe_service.convert import region_array as _region_array
from doe_service.schemas.design import (
    CandidatesResponse,
    DesignDocument,
    DesignResponse,
    OptimalDesignResponse,
    SearchReport,
    ValidateResponse,
)
from doe_service.schemas.factors import factor_schema_to_factor
from doe_service.schemas.generation import (
    AugmentRequest,
    BoxBehnkenRequest,
    CandidatesRequest,
    CentralCompositeRequest,
    DefinitiveScreeningRequest,
    ExtremeVerticesRequest,
    FactorsRequest,
    FractionalFactorialRequest,
    FullFactorialRequest,
    OptimalRequest,
    PlackettBurmanRequest,
    SimplexCentroidRequest,
    SimplexLatticeRequest,
    SpaceFillingRequest,
)
from doe_service.schemas.operations import (
    ProjectRequest,
    RandomizeRequest,
    ReplicateRequest,
    ResponsesRequest,
    ValidateRequest,
)

router = APIRouter(prefix="/v1/designs", tags=["designs"])


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _factors(body: FactorsRequest) -> list[Factor]:
    """Convert a request's ``factors: list[FactorSchema]`` to real ``doe`` factors.

    Every generation/candidates endpoint funnels through here, so this is the one
    shared prologue where ``max_factors`` is enforced for all of them at once
    (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).
    """
    factors = [factor_schema_to_factor(f) for f in body.factors]
    check_factor_count(factors)
    return factors


def _design_response(design: Design, warnings: list[str]) -> DesignResponse:
    """``max_runs`` is checked here too -- every generation/operation endpoint that
    returns a plain ``DesignResponse`` (i.e. every one except ``/optimal``/``/augment``,
    which check it explicitly around their own search) shares this response builder."""
    check_run_count(design.n_runs)
    return DesignResponse(
        design=DesignDocument.model_validate(design.to_dict()), warnings=warnings
    )


def _finite(value: float) -> float | None:
    """Non-finite (``-inf`` from a degenerate/rank-deficient search) -> ``None``, per the
    "JSON has no NaN/Infinity" rule (``docs/WEBSERVICE_API.md`` "Conventions")."""
    return value if math.isfinite(value) else None


def _search_report(result: OptimalDesign) -> SearchReport:
    """The top-level search report from a fresh ``coordinate_exchange`` call."""
    return SearchReport(
        criterion=result.criterion,
        score=_finite(result.score),
        d_efficiency=result.d_efficiency,
        n_restarts=result.n_restarts,
        converged=result.converged,
    )


def _search_report_from_meta(meta: Mapping[str, object]) -> SearchReport:
    """The top-level search report reconstructed from an augmented design's ``meta``.

    ``doe.augment`` builds on ``coordinate_exchange`` internally and its returned
    design's ``meta`` already carries the full search report (criterion, score,
    d_efficiency, n_restarts, converged) -- pulled out here rather than re-deriving
    fixed rows by hand or calling ``coordinate_exchange`` a second time.
    """
    return SearchReport(
        criterion=cast(str, meta["criterion"]),
        score=_finite(cast(float, meta["score"])),
        d_efficiency=cast(float, meta["d_efficiency"]),
        n_restarts=cast(int, meta["n_restarts"]),
        converged=cast(bool, meta["converged"]),
    )


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


@router.post("/full-factorial")
def full_factorial(body: FullFactorialRequest) -> DesignResponse:
    """Wraps ``doe.full_factorial``."""

    def run() -> Design:
        return _full_factorial(_factors(body), levels=body.levels)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/fractional-factorial")
def fractional_factorial(body: FractionalFactorialRequest) -> DesignResponse:
    """Wraps ``doe.fractional_factorial``."""

    def run() -> Design:
        return _fractional_factorial(_factors(body), generators=body.generators)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/plackett-burman")
def plackett_burman(body: PlackettBurmanRequest) -> DesignResponse:
    """Wraps ``doe.plackett_burman``."""

    def run() -> Design:
        return _plackett_burman(_factors(body))

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/definitive-screening")
def definitive_screening(body: DefinitiveScreeningRequest) -> DesignResponse:
    """Wraps ``doe.definitive_screening``."""

    def run() -> Design:
        return _definitive_screening(
            _factors(body),
            extra_center_runs=body.extra_center_runs,
            fake_factors=body.fake_factors,
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/central-composite")
def central_composite(body: CentralCompositeRequest) -> DesignResponse:
    """Wraps ``doe.central_composite``."""

    def run() -> Design:
        return _central_composite(
            _factors(body), alpha=body.alpha, center=body.center, fraction=body.fraction
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/box-behnken")
def box_behnken(body: BoxBehnkenRequest) -> DesignResponse:
    """Wraps ``doe.box_behnken``."""

    def run() -> Design:
        return _box_behnken(_factors(body), center=body.center)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/space-filling")
def space_filling(body: SpaceFillingRequest) -> DesignResponse:
    """Wraps ``doe.latin_hypercube`` / ``doe.sobol`` / ``doe.halton`` via ``sampler``."""

    def run() -> Design:
        factors = _factors(body)
        if body.sampler == "lhs":
            return _latin_hypercube(
                factors, n_runs=body.n_runs, criterion=body.criterion, seed=body.seed
            )
        if body.sampler == "sobol":
            return _sobol(factors, n_runs=body.n_runs, scramble=body.scramble, seed=body.seed)
        return _halton(factors, n_runs=body.n_runs, scramble=body.scramble, seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/simplex-lattice")
def simplex_lattice(body: SimplexLatticeRequest) -> DesignResponse:
    """Wraps ``doe.simplex_lattice``."""

    def run() -> Design:
        return _simplex_lattice(_factors(body), degree=body.degree)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/simplex-centroid")
def simplex_centroid(body: SimplexCentroidRequest) -> DesignResponse:
    """Wraps ``doe.simplex_centroid``."""

    def run() -> Design:
        return _simplex_centroid(_factors(body))

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/extreme-vertices")
def extreme_vertices(body: ExtremeVerticesRequest) -> DesignResponse:
    """Wraps ``doe.extreme_vertices``."""

    def run() -> Design:
        return _extreme_vertices(_factors(body), include_centroid=body.include_centroid)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/optimal")
def optimal(body: OptimalRequest) -> OptimalDesignResponse:
    """Wraps ``doe.coordinate_exchange`` directly (not the ``d_optimal``/``i_optimal``
    wrappers), so the response can carry the ``OptimalDesign`` search report at the top
    level alongside the design.

    ``n_runs`` and the ``n_restarts``/``max_iter`` search budget are capped before the
    (expensive) exchange search ever runs, not after -- the whole point of the cap is to
    keep the search itself bounded (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).
    """
    check_run_count(body.n_runs)
    check_search_budget(body.n_restarts, body.max_iter)

    def run() -> OptimalDesign:
        factors = _factors(body)
        region = _region_array(body.region, n_factors=len(factors))
        return _coordinate_exchange(
            factors,
            n_runs=body.n_runs,
            model=body.model,
            criterion=body.criterion,
            region=region,
            n_restarts=body.n_restarts,
            max_iter=body.max_iter,
            seed=body.seed,
        )

    with captured_warnings() as warns:
        result = call_library(run)
    search = _search_report(result)
    return OptimalDesignResponse(
        design=DesignDocument.model_validate(result.design.to_dict()),
        search=search,
        warnings=warns,
    )


@router.post("/augment")
def augment(body: AugmentRequest) -> OptimalDesignResponse:
    """Wraps ``doe.augment`` -- holds the posted design's rows fixed
    (``point_type="existing"``) and searches only the added rows, exactly as
    ``doe.augment`` does. ``doe.augment`` builds on ``coordinate_exchange`` internally
    and its returned design's ``meta`` already carries the full search report (criterion,
    score, d_efficiency, n_restarts, converged) -- pulled out here for the top-level
    ``search`` field rather than re-deriving fixed rows by hand.

    ``n_runs`` (the number of rows being *added*) and the search budget are capped
    before the search runs; the existing design's own factor/run counts are already
    capped inside :func:`~doe_service.convert.design_from_document`, and the augmented
    *total* is capped again once the search returns, since existing + new can exceed the
    per-design cap even when each half is within it.
    """
    check_run_count(body.n_runs)
    check_search_budget(body.n_restarts, body.max_iter)

    def run() -> Design:
        design = design_from_document(body.design.model_dump())
        region = _region_array(body.region, n_factors=len(design.factors))
        return _augment(
            design,
            n_runs=body.n_runs,
            model=body.model,
            criterion=body.criterion,
            seed=body.seed,
            region=region,
            n_restarts=body.n_restarts,
            max_iter=body.max_iter,
        )

    with captured_warnings() as warns:
        design = call_library(run)
    check_run_count(design.n_runs)
    search = _search_report_from_meta(design.meta)
    return OptimalDesignResponse(
        design=DesignDocument.model_validate(design.to_dict()), search=search, warnings=warns
    )


@router.post("/candidates")
def candidates(body: CandidatesRequest) -> CandidatesResponse:
    """Wraps ``doe.candidate_grid`` / ``doe.mixture_candidates`` (dispatch on factor set)."""

    def run() -> tuple[np.ndarray, Literal["grid", "mixture"]]:
        factors = _factors(body)
        fs = FactorSet(factors)
        if fs.is_mixture:
            resolution = body.resolution if body.resolution is not None else 10
            return _mixture_candidates(factors, resolution=resolution), "mixture"
        levels = body.levels if body.levels is not None else 3
        return _candidate_grid(factors, levels=levels), "grid"

    with captured_warnings():
        points, kind = call_library(run)
    points_list = cast(list[list[float]], jsonable(points.tolist()))
    return CandidatesResponse(points=points_list, kind=kind)


# --------------------------------------------------------------------------- #
# operations
# --------------------------------------------------------------------------- #


@router.post("/validate")
def validate(body: ValidateRequest) -> ValidateResponse:
    """Wraps ``doe.validate_design_dict`` — 200 with ``{valid, errors}`` either way."""
    try:
        _validate_design_dict(body.design, check_ranges=body.check_ranges)
    except ValidationError as exc:
        return ValidateResponse(valid=False, errors=list(exc.errors))
    return ValidateResponse(valid=True, errors=[])


@router.post("/randomize")
def randomize(body: RandomizeRequest) -> DesignResponse:
    """Wraps ``Design.randomize``."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump())
        return design.randomize(seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/replicate")
def replicate(body: ReplicateRequest) -> DesignResponse:
    """Wraps ``Design.replicate``."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump())
        return design.replicate(body.n, each=body.each)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/project")
def project(body: ProjectRequest) -> DesignResponse:
    """Wraps ``Design.project`` — the post-screening "keep the survivors" step."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump())
        return design.project(body.factors)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)


@router.post("/responses")
def responses(body: ResponsesRequest) -> DesignResponse:
    """Wraps ``Design.with_responses`` — attach readouts as aligned columns."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump())
        return design.with_responses(**body.responses)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns)
