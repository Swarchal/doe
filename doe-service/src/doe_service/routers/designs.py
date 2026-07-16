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
from typing import Annotated, Literal, cast

import numpy as np
from fastapi import APIRouter, Depends

from doe import (
    Design,
    Factor,
    FactorSet,
    OptimalDesign,
    ValidationError,
)
from doe import augment as _augment
from doe import blocked_factorial as _blocked_factorial
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
from doe import latin_square as _latin_square
from doe import mixture_candidates as _mixture_candidates
from doe import plackett_burman as _plackett_burman
from doe import randomized_complete_block as _randomized_complete_block
from doe import simplex_centroid as _simplex_centroid
from doe import simplex_lattice as _simplex_lattice
from doe import sobol as _sobol
from doe import split_plot as _split_plot
from doe import validate_design_dict as _validate_design_dict
from doe_service.convert import (
    call_library,
    captured_warnings,
    check_factor_count,
    check_projected_runs,
    check_run_count,
    check_search_budget,
    design_from_document,
    full_factorial_runs,
    jsonable,
)
from doe_service.convert import region_array as _region_array
from doe_service.deps import app_limits
from doe_service.limits import Limits
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
    BlockedFactorialRequest,
    BoxBehnkenRequest,
    CandidatesRequest,
    CentralCompositeRequest,
    DefinitiveScreeningRequest,
    ExtremeVerticesRequest,
    FactorsRequest,
    FractionalFactorialRequest,
    FullFactorialRequest,
    LatinSquareRequest,
    OptimalRequest,
    PlackettBurmanRequest,
    RandomizedCompleteBlockRequest,
    SimplexCentroidRequest,
    SimplexLatticeRequest,
    SpaceFillingRequest,
    SplitPlotRequest,
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


def _factors(body: FactorsRequest, limits: Limits) -> list[Factor]:
    """Convert a request's ``factors: list[FactorSchema]`` to real ``doe`` factors.

    Every generation/candidates endpoint funnels through here, so this is the one
    shared prologue where ``max_factors`` is enforced for all of them at once
    (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).
    """
    factors = [factor_schema_to_factor(f) for f in body.factors]
    check_factor_count(factors, limits=limits)
    return factors


def _design_response(design: Design, warnings: list[str], limits: Limits) -> DesignResponse:
    """``max_runs`` is checked here too -- every generation/operation endpoint that
    returns a plain ``DesignResponse`` (i.e. every one except ``/optimal``/``/augment``,
    which check it explicitly around their own search) shares this response builder."""
    check_run_count(design.n_runs, limits=limits)
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
def full_factorial(
    body: FullFactorialRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.full_factorial``.

    The run count is the *product* of the per-factor levels, so it is projected and capped
    before generating: ``max_factors`` (32) and an unbounded ``levels`` otherwise admit
    requests that exhaust memory long before ``_design_response``'s ``max_runs`` check runs.
    """

    def run() -> Design:
        factors = _factors(body, limits)
        check_projected_runs(
            full_factorial_runs(factors, body.levels),
            what="this factor/level combination",
            limits=limits,
        )
        return _full_factorial(factors, levels=body.levels)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/fractional-factorial")
def fractional_factorial(
    body: FractionalFactorialRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.fractional_factorial``."""

    def run() -> Design:
        factors = _factors(body, limits)
        # the fraction is the 2**n_base factorial of the base factors (the generated ones are
        # products of those columns, adding no rows)
        n_base = len(factors) - len(body.generators)
        if n_base >= 1:
            check_projected_runs(2**n_base, what=f"a 2**{n_base} base factorial", limits=limits)
        return _fractional_factorial(factors, generators=body.generators)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/plackett-burman")
def plackett_burman(
    body: PlackettBurmanRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.plackett_burman``."""

    def run() -> Design:
        return _plackett_burman(_factors(body, limits))

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/definitive-screening")
def definitive_screening(
    body: DefinitiveScreeningRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.definitive_screening``."""

    def run() -> Design:
        return _definitive_screening(
            _factors(body, limits),
            extra_center_runs=body.extra_center_runs,
            fake_factors=body.fake_factors,
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/central-composite")
def central_composite(
    body: CentralCompositeRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.central_composite``."""

    def run() -> Design:
        # `center` replicates are appended row-for-row, so it alone can exceed the cap; the
        # factorial core is bounded by max_factors unless a fraction shrinks it further
        factors = _factors(body, limits)
        check_projected_runs(body.center, what=f"center={body.center}", limits=limits)
        if body.fraction is None:
            check_projected_runs(
                full_factorial_runs(factors, 2),
                what="this factor count's factorial core",
                limits=limits,
            )
        return _central_composite(
            factors, alpha=body.alpha, center=body.center, fraction=body.fraction
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/box-behnken")
def box_behnken(
    body: BoxBehnkenRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.box_behnken``."""

    def run() -> Design:
        # `center` replicates are appended row-for-row, so it alone can exceed the cap
        check_projected_runs(body.center, what=f"center={body.center}", limits=limits)
        return _box_behnken(_factors(body, limits), center=body.center)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/space-filling")
def space_filling(
    body: SpaceFillingRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.latin_hypercube`` / ``doe.sobol`` / ``doe.halton`` via ``sampler``."""

    def run() -> Design:
        factors = _factors(body, limits)
        # a space-filling sampler materialises exactly ``n_runs`` rows (and the default
        # maximin criterion additionally scores 30 candidate hypercubes with an O(n**2)
        # pdist), so the run count must be capped *before* the sampler runs -- ``n_runs`` is
        # an unbounded int on the wire. Mirrors ``/optimal``'s pre-flight ``check_run_count``.
        check_run_count(body.n_runs, limits=limits)
        if body.sampler == "lhs":
            return _latin_hypercube(
                factors, n_runs=body.n_runs, criterion=body.criterion, seed=body.seed
            )
        if body.sampler == "sobol":
            return _sobol(factors, n_runs=body.n_runs, scramble=body.scramble, seed=body.seed)
        return _halton(factors, n_runs=body.n_runs, scramble=body.scramble, seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/simplex-lattice")
def simplex_lattice(
    body: SimplexLatticeRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.simplex_lattice``."""

    def run() -> Design:
        factors = _factors(body, limits)
        # a {k, m} lattice has C(k + m - 1, m) blends -- combinatorial in the degree
        if body.degree >= 0:
            check_projected_runs(
                math.comb(len(factors) + body.degree - 1, body.degree),
                what="this component/degree combination",
                limits=limits,
            )
        return _simplex_lattice(factors, degree=body.degree)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/simplex-centroid")
def simplex_centroid(
    body: SimplexCentroidRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.simplex_centroid``."""

    def run() -> Design:
        return _simplex_centroid(_factors(body, limits))

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/extreme-vertices")
def extreme_vertices(
    body: ExtremeVerticesRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.extreme_vertices``."""

    def run() -> Design:
        return _extreme_vertices(_factors(body, limits), include_centroid=body.include_centroid)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/split-plot")
def split_plot(
    body: SplitPlotRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.split_plot`` -- factors flagged ``hard_to_change`` form the whole-plot
    stratum, the rest the sub-plot stratum. ``whole_plot_design``/``sub_plot_design`` are
    ``"full"`` or a design document on exactly that stratum's factors (converted here the
    same way any posted design is: validated + factor/run capped)."""

    def run() -> Design:
        factors = _factors(body, limits)
        whole = (
            body.whole_plot_design
            if isinstance(body.whole_plot_design, str)
            else design_from_document(body.whole_plot_design.model_dump(), limits=limits)
        )
        sub = (
            body.sub_plot_design
            if isinstance(body.sub_plot_design, str)
            else design_from_document(body.sub_plot_design.model_dump(), limits=limits)
        )
        # the split-plot design crosses the two strata: reps x whole-plot rows x sub-plot rows,
        # and a "full" stratum is the 2**k factorial of its factors
        hard = [f for f in factors if getattr(f, "hard_to_change", False)]
        easy = [f for f in factors if not getattr(f, "hard_to_change", False)]
        n_whole = full_factorial_runs(hard, 2) if isinstance(whole, str) else whole.n_runs
        n_sub = full_factorial_runs(easy, 2) if isinstance(sub, str) else sub.n_runs
        if body.n_whole_plot_reps >= 0:
            check_projected_runs(
                body.n_whole_plot_reps * n_whole * n_sub,
                what=(
                    f"{body.n_whole_plot_reps} rep(s) x {n_whole} whole plot(s) "
                    f"x {n_sub} sub-plot run(s)"
                ),
                limits=limits,
            )
        return _split_plot(
            factors,
            whole_plot_design=whole,
            sub_plot_design=sub,
            n_whole_plot_reps=body.n_whole_plot_reps,
            seed=body.seed,
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/randomized-complete-block")
def randomized_complete_block(
    body: RandomizedCompleteBlockRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.randomized_complete_block``. Treatments are a factor list or an integer
    ``n_treatments`` -- exactly one; supplying neither or both is a 422 ``infeasible``."""

    def run() -> Design:
        if (body.factors is None) == (body.n_treatments is None):
            raise ValueError("provide exactly one of 'factors' or 'n_treatments'")
        treatments: list[Factor] | int
        if body.factors is not None:
            treatments = [factor_schema_to_factor(f) for f in body.factors]
            check_factor_count(treatments, limits=limits)
            n_treatments = full_factorial_runs(treatments, 2)
        else:
            treatments = cast(int, body.n_treatments)
            n_treatments = treatments
        # every treatment is run once per block: t * b runs, both otherwise unbounded
        if n_treatments >= 0 and body.n_blocks >= 0:
            check_projected_runs(
                n_treatments * body.n_blocks,
                what=f"{n_treatments} treatment(s) x {body.n_blocks} block(s)",
                limits=limits,
            )
        return _randomized_complete_block(treatments, n_blocks=body.n_blocks, seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/latin-square")
def latin_square(
    body: LatinSquareRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.latin_square`` -- a ``treatments x treatments`` square design."""

    def run() -> Design:
        # a t x t square is t**2 runs, and `treatments` is otherwise unbounded
        if body.treatments >= 0:
            check_projected_runs(
                body.treatments**2, what=f"treatments={body.treatments}", limits=limits
            )
        return _latin_square(body.treatments, seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/blocked-factorial")
def blocked_factorial(
    body: BlockedFactorialRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``doe.blocked_factorial`` -- a ``2^k`` factorial confounded into blocks via
    the defining ``block_generators``."""

    def run() -> Design:
        factors = _factors(body, limits)
        # blocking splits a full 2**k factorial; it does not shrink it
        check_projected_runs(
            full_factorial_runs(factors, 2),
            what="this factor count's 2**k factorial",
            limits=limits,
        )
        return _blocked_factorial(
            factors, block_generators=body.block_generators, seed=body.seed
        )

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/optimal")
def optimal(
    body: OptimalRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> OptimalDesignResponse:
    """Wraps ``doe.coordinate_exchange`` directly (not the ``d_optimal``/``i_optimal``
    wrappers), so the response can carry the ``OptimalDesign`` search report at the top
    level alongside the design.

    ``n_runs`` and the ``n_restarts``/``max_iter`` search budget are capped before the
    (expensive) exchange search ever runs, not after -- the whole point of the cap is to
    keep the search itself bounded (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).

    A large search may run its restarts in parallel: the server (never the client) picks
    ``n_jobs`` from the deployment's :class:`~doe_service.limits.Limits`
    (:meth:`~doe_service.limits.Limits.optimal_n_jobs`), which is disabled by default and
    enabled by front-ends like doe-web.
    """
    check_run_count(body.n_runs, limits=limits)
    check_search_budget(body.n_restarts, body.max_iter, limits=limits)
    n_jobs = limits.optimal_n_jobs(body.n_runs)

    def run() -> OptimalDesign:
        factors = _factors(body, limits)
        region = _region_array(body.region, n_factors=len(factors), limits=limits)
        return _coordinate_exchange(
            factors,
            n_runs=body.n_runs,
            model=body.model,
            criterion=body.criterion,
            region=region,
            n_restarts=body.n_restarts,
            max_iter=body.max_iter,
            seed=body.seed,
            n_jobs=n_jobs,
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
def augment(
    body: AugmentRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> OptimalDesignResponse:
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
    check_run_count(body.n_runs, limits=limits)
    check_search_budget(body.n_restarts, body.max_iter, limits=limits)

    def run() -> Design:
        design = design_from_document(body.design.model_dump(), limits=limits)
        region = _region_array(body.region, n_factors=len(design.factors), limits=limits)
        # threshold on the *augmented total* -- the search operates over all rows, holding the
        # existing ones fixed while it exchanges only the added ones.
        n_jobs = limits.optimal_n_jobs(design.n_runs + body.n_runs)
        return _augment(
            design,
            n_runs=body.n_runs,
            model=body.model,
            criterion=body.criterion,
            seed=body.seed,
            region=region,
            n_restarts=body.n_restarts,
            max_iter=body.max_iter,
            n_jobs=n_jobs,
        )

    with captured_warnings() as warns:
        design = call_library(run)
    check_run_count(design.n_runs, limits=limits)
    search = _search_report_from_meta(design.meta)
    return OptimalDesignResponse(
        design=DesignDocument.model_validate(design.to_dict()), search=search, warnings=warns
    )


@router.post("/candidates")
def candidates(
    body: CandidatesRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> CandidatesResponse:
    """Wraps ``doe.candidate_grid`` / ``doe.mixture_candidates`` (dispatch on factor set)."""

    def run() -> tuple[np.ndarray, Literal["grid", "mixture"]]:
        factors = _factors(body, limits)
        fs = FactorSet(factors)
        if fs.is_mixture:
            resolution = body.resolution if body.resolution is not None else 10
            # a simplex lattice of `resolution` steps over k components has C(res + k - 1, k - 1)
            # points -- combinatorial in both, so project before building (see /full-factorial)
            if resolution >= 0:
                check_projected_runs(
                    math.comb(resolution + len(factors) - 1, len(factors) - 1),
                    what="this component/resolution combination",
                    limits=limits,
                )
            return _mixture_candidates(factors, resolution=resolution), "mixture"
        levels = body.levels if body.levels is not None else 3
        check_projected_runs(
            full_factorial_runs(factors, levels),
            what="this factor/level combination",
            limits=limits,
        )
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
def randomize(
    body: RandomizeRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``Design.randomize``."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump(), limits=limits)
        return design.randomize(seed=body.seed)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/replicate")
def replicate(
    body: ReplicateRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``Design.replicate``."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump(), limits=limits)
        # replication multiplies the posted design's rows, and `n` is otherwise unbounded
        if body.n >= 0:
            check_projected_runs(
                body.n * design.n_runs,
                what=f"{body.n} replicate(s) of a {design.n_runs}-run design",
                limits=limits,
            )
        return design.replicate(body.n, each=body.each)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/project")
def project(
    body: ProjectRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``Design.project`` — the post-screening "keep the survivors" step."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump(), limits=limits)
        return design.project(body.factors)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)


@router.post("/responses")
def responses(
    body: ResponsesRequest, limits: Annotated[Limits, Depends(app_limits)]
) -> DesignResponse:
    """Wraps ``Design.with_responses`` — attach readouts as aligned columns."""

    def run() -> Design:
        design = design_from_document(body.design.model_dump(), limits=limits)
        return design.with_responses(**body.responses)

    with captured_warnings() as warns:
        design = call_library(run)
    return _design_response(design, warns, limits)
