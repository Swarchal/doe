"""Request → library plumbing (Milestone 1, ``docs/WEBSERVICE_BUILD.md`` §1.2).

The four seams between the wire format and ``doe``. Routers call these and nothing
else for conversion, so the rules (validate-then-build, model resolution, warning
mapping, NaN policy) live in exactly one place.
"""

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

from doe import (
    MODEL_SPECS,
    CategoricalFactor,
    Design,
    Factor,
    SaturatedFitWarning,
    ValidationError,
    validate_design_dict,
)
from doe.serialization import json_safe
from doe_service.errors import Infeasible
from doe_service.limits import DEFAULT_LIMITS, LimitExceeded, Limits
from doe_service.schemas.common import ModelSpecObject

#: A model spec as it arrives on the wire: a convenience name, ``{order, interactions}``
#: mapping, or an already-parsed ``ModelSpecObject``.
ModelSpecLike = str | Mapping[str, Any] | ModelSpecObject


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# limits (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6)
# --------------------------------------------------------------------------- #
#
# One shared validator per cap rather than scattering ``if`` statements through the
# routers. Every check names the cap and the ceiling in its message, per the spec. Each
# takes the deployment's ``Limits`` (defaulting to ``DEFAULT_LIMITS`` only when called
# outside a request); routers obtain it from the ``app_limits`` FastAPI dependency
# (``deps.py``) and thread it in, so a ``create_app(limits=...)`` override is honoured by
# every cap -- not only the body-size middleware and the parallelism policy.


def check_factor_count(factors: Sequence[Factor], *, limits: Limits = DEFAULT_LIMITS) -> None:
    """422 ``limit_exceeded`` when a factor list exceeds ``limits.max_factors``.

    Guards model-matrix width (``docs/WEBSERVICE_API.md`` "Limits"). Called from the
    designs router's ``_factors`` helper -- the single place every generation/candidates
    endpoint converts wire factors to real ``doe`` factors -- and from
    :func:`design_from_document` for every endpoint that instead posts a full design.
    """
    if len(factors) > limits.max_factors:
        raise LimitExceeded(
            f"too many factors: {len(factors)} exceeds the cap of {limits.max_factors}"
        )


def check_run_count(n_runs: int, *, limits: Limits = DEFAULT_LIMITS) -> None:
    """422 ``limit_exceeded`` when a design's run count exceeds ``limits.max_runs``.

    Guards payload size and OLS cost (``docs/WEBSERVICE_API.md`` "Limits"). Called from
    :func:`design_from_document` (every posted design) and from the designs router's
    ``_design_response`` helper (every generated/transformed design) plus, pre-flight,
    wherever a request names its run count directly (``/optimal``'s ``n_runs``).
    """
    if n_runs > limits.max_runs:
        raise LimitExceeded(f"too many runs: {n_runs} exceeds the cap of {limits.max_runs}")


def check_projected_runs(n_runs: int, *, what: str, limits: Limits = DEFAULT_LIMITS) -> None:
    """422 ``limit_exceeded`` for a run count *predicted from the request*, before generating.

    :func:`check_run_count` runs on a design that already exists, which is too late for the
    generators whose run count is exponential in their parameters: ``full_factorial`` over 8
    factors at 10 levels is 10**8 runs, and the process dies allocating them long before any
    cap is consulted. The routes that can blow up this way project their run count from the
    request first and call this, so the cap is enforced against the *intended* design rather
    than the materialized one. ``what`` names the parameters that drove the projection, so the
    caller learns which knob to turn down.
    """
    if n_runs > limits.max_runs:
        raise LimitExceeded(
            f"{what} would produce {n_runs} runs, exceeding the cap of {limits.max_runs}"
        )


def full_factorial_runs(factors: Sequence[Factor], levels: int | Sequence[int]) -> int:
    """Rows a ``full_factorial`` over these factors/levels would generate (the level product).

    Mirrors ``doe.full_factorial``'s own level resolution: a categorical factor always
    contributes its own number of levels, whatever ``levels`` says. A malformed ``levels``
    (wrong length, or a count below 2) projects as ``0`` so the request falls through to the
    library, whose error message for that is the right one -- this helper exists to catch
    designs that are *too big*, not to re-validate shape.
    """
    per: list[int] = [levels] * len(factors) if isinstance(levels, int) else list(levels)
    if len(per) != len(factors):
        return 0
    total = 1
    for factor, requested in zip(factors, per, strict=True):
        count = len(factor.levels) if isinstance(factor, CategoricalFactor) else int(requested)
        if count < 2:
            return 0
        total *= count
    return total


def check_search_budget(
    n_restarts: int, max_iter: int, *, limits: Limits = DEFAULT_LIMITS
) -> None:
    """422 ``limit_exceeded`` for a coordinate-exchange search budget above the caps.

    The spec names a combined ``n_restarts x max_iter`` wall-time cap (default
    ``20 x 100``, ``docs/WEBSERVICE_API.md`` "Limits"); ``n_restarts``/``max_iter`` are
    checked individually against ``limits.max_restarts``/``limits.max_iter``, which
    together bound that product. Used by ``/v1/designs/optimal`` and
    ``/v1/designs/augment``, the only two endpoints that drive ``coordinate_exchange``.
    """
    if n_restarts > limits.max_restarts:
        raise LimitExceeded(f"n_restarts {n_restarts} exceeds the cap of {limits.max_restarts}")
    if max_iter > limits.max_iter:
        raise LimitExceeded(f"max_iter {max_iter} exceeds the cap of {limits.max_iter}")


def region_array(
    region: Sequence[Sequence[float]] | None,
    *,
    n_factors: int,
    limits: Limits = DEFAULT_LIMITS,
) -> np.ndarray | None:
    """``region`` arrives as ``list[list[float]]``: cap its row count, shape-check each
    row against ``n_factors``, then build the ndarray the library expects.

    Shared by ``/v1/designs/optimal``, ``/v1/designs/augment``, and
    ``/v1/analysis/diagnostics`` -- the three endpoints that take an explicit coded
    candidate/region array on the wire (``docs/WEBSERVICE_API.md`` "region" convention).
    """
    if region is None:
        return None
    if len(region) > limits.max_region_rows:
        raise LimitExceeded(
            f"region has {len(region)} rows, exceeding the cap of {limits.max_region_rows}"
        )
    bad = [i for i, row in enumerate(region) if len(row) != n_factors]
    if bad:
        raise Infeasible(
            f"region must have {n_factors} coordinate(s) per row (one per factor); "
            f"row(s) {bad} do not match"
        )
    return np.asarray(region, dtype=float)


def design_from_document(
    document: Mapping[str, Any], *, limits: Limits = DEFAULT_LIMITS
) -> Design:
    """``validate_design_dict`` -> ``Design.from_dict``, then the factor/run count caps.

    ``doe.ValidationError`` propagates; the M1 handlers map it to the 422 envelope with
    every collected problem. Deliberately *not* run through :func:`call_library` — a
    ``ValidationError`` carries its own precise ``.errors`` list and must not be
    collapsed into the generic ``infeasible`` message.

    Every endpoint that takes a posted ``design`` document (design operations, analysis,
    optimize, plot-data) funnels through here, so this is the one shared prologue where
    ``max_factors``/``max_runs`` are enforced for all of them at once (Milestone 6,
    ``docs/WEBSERVICE_BUILD.md`` §6). ``limits`` is the deployment's configured caps,
    threaded in from the router's ``app_limits`` dependency.
    """
    validate_design_dict(document)
    design = Design.from_dict(document)
    check_factor_count(list(design.factors), limits=limits)
    check_run_count(design.n_runs, limits=limits)
    return design


def _resolve_one(spec: ModelSpecLike) -> tuple[int, bool]:
    if isinstance(spec, ModelSpecObject):
        return spec.order, spec.interactions
    if isinstance(spec, str):
        try:
            return MODEL_SPECS[spec]
        except KeyError:
            raise ValueError(
                f"unknown model {spec!r}; expected one of {sorted(MODEL_SPECS)} "
                "or {'order': 1 | 2, 'interactions': bool}"
            ) from None
    if isinstance(spec, Mapping):
        if "order" not in spec:
            raise ValueError("model object must include 'order' (1 or 2)")
        order = spec["order"]
        if order not in (1, 2):
            raise ValueError(f"model 'order' must be 1 or 2, got {order!r}")
        interactions = bool(spec.get("interactions", True))
        return int(order), interactions
    raise ValueError(f"unrecognized model spec: {spec!r}")


def resolve_model(model: ModelSpecLike | None, *, default: ModelSpecLike) -> tuple[int, bool]:
    """Resolve a wire model spec to ``(order, interactions)``.

    ``"linear"``/``"quadratic"``/``"scheffe-linear"``/``"scheffe-quadratic"`` or an
    ``{order, interactions}`` object; ``model=None`` falls back to ``default``. Scheffé
    names additionally require an all-mixture design at the call site (the library
    enforces it via ``fit_ols(..., model=...)``; the resulting ``ValueError`` maps to
    422 ``infeasible`` when routed through :func:`call_library`).
    """
    return _resolve_one(model if model is not None else default)


@contextmanager
def captured_warnings() -> Iterator[list[str]]:
    """Record library warnings as the spec's warning strings.

    Categories map by class -- ``doe.SaturatedFitWarning`` -> ``"saturated_model"`` (a
    Milestone 0 library addition); unknown warnings fall back to ``str(message)``.
    """
    collected: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield collected
        for w in caught:
            if issubclass(w.category, SaturatedFitWarning):
                collected.append("saturated_model")
            else:
                collected.append(str(w.message))


def jsonable(value: object) -> object:
    """NaN/Inf -> ``None``, numpy -> native -- the boundary safety net.

    Thin wrapper over ``doe.serialization.json_safe`` (Milestone 0) for values that did
    not come from a library ``to_dict`` (e.g. plot meshes).
    """
    return json_safe(value)


def call_library(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call a library function, mapping a bare ``ValueError`` to 422 ``infeasible``.

    Routers use this for every ``doe`` call whose ``ValueError`` should become an
    ``infeasible`` response (a domain message: bad Sobol run count, infeasible mixture
    bounds, mixed mixture/non-mixture factor sets, bad desirability brackets, ...)
    rather than a fake ``validation_error`` or an opaque 500. Scoped deliberately: a
    genuine service bug (a ``TypeError``/``KeyError`` from a router mis-calling the
    library) is *not* caught here, so it still surfaces as 500 ``internal`` instead of a
    misleading 422. ``doe.ValidationError`` (a ``ValueError`` subclass, but with its own
    precise ``.errors`` list) passes through unchanged rather than being collapsed into
    the generic ``infeasible`` message.
    """
    try:
        return fn(*args, **kwargs)
    except ValidationError:
        raise
    except ValueError as exc:
        raise Infeasible(str(exc)) from exc
