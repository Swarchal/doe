"""Optimal (computer-generated) designs via coordinate exchange (Phase 3b).

Where the named recipes (factorial, CCD, Box-Behnken) produce a fixed run set for a fixed
model, the coordinate-exchange engine here *builds* a run set to maximise a chosen criterion
(D- or I-optimality) for a user-specified model, run budget, and candidate region. This is
what handles irregular constraints, custom models, odd run counts, and augmenting an existing
design -- the cases the recipes can't.

The engine's objective is a diagnostic: ``criterion="D"`` maximises
:func:`~doe.analysis.diagnostics.log_det_information`; ``criterion="I"`` minimises the average
prediction variance over the region. Generators return a plain :class:`~doe.design.Design`
(so ``coded()``/fitting/ANOVA/plots work unchanged) with the search diagnostics stashed in
``design.meta``. See ``docs/PHASE3.md`` for the algorithm and correctness anchors.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..analysis.diagnostics import log_det_information
from ..analysis.model import coded_design_points, expand_coded_points
from ..design import Design, _draw_seed
from ..factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet

Criterion = Literal["D", "I"]
ModelSpec = Literal["linear", "quadratic"]

_MODEL_SPECS: dict[str, tuple[int, bool]] = {
    "linear": (1, True),
    "quadratic": (2, True),
}


def _model_spec(model: str) -> tuple[int, bool]:
    """Normalize an optimal-design model name to ``(order, interactions)``."""
    if model not in _MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(_MODEL_SPECS)}")
    return _MODEL_SPECS[model]


# --------------------------------------------------------------------------- #
# 2.1 Candidate region
# --------------------------------------------------------------------------- #


def candidate_grid(factors: Sequence[Factor], *, levels: int = 3) -> np.ndarray:
    """A discrete candidate set: the full ``levels``-level grid over the coded box.

    Returns an ``(n_candidates, k)`` array of coded points. Continuous factors take
    ``levels`` equally-spaced values in ``[-1, +1]``; categorical factors take their discrete
    contrast levels. Supply a hand-built array instead to encode constraints (just omit the
    infeasible points). The region is shared with the G/I-efficiency sampling in
    :mod:`doe.analysis.diagnostics`.
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")

    axes: list[np.ndarray] = []
    for factor in factors:
        if isinstance(factor, ContinuousFactor):
            axes.append(np.linspace(-1.0, 1.0, levels))
        elif isinstance(factor, CategoricalFactor):
            axes.append(np.linspace(-1.0, 1.0, len(factor.levels)))
        else:  # pragma: no cover - Factor is currently exhaustive
            raise TypeError(f"unsupported factor type for {factor!r}")

    if not axes:
        return np.empty((1, 0), dtype=float)
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def _validate_region(region: np.ndarray, *, n_factors: int) -> np.ndarray:
    region = np.asarray(region, dtype=float)
    if region.ndim != 2:
        raise ValueError("region must be a 2-D array with shape (n_candidates, n_factors)")
    if region.shape[1] != n_factors:
        raise ValueError(
            f"region has {region.shape[1]} columns but factors has {n_factors} entries"
        )
    if region.shape[0] == 0:
        raise ValueError("region must contain at least one candidate point")
    if not np.all(np.isfinite(region)):
        raise ValueError("region must contain only finite coded values")
    if np.any((region < -1.0 - 1e-12) | (region > 1.0 + 1e-12)):
        raise ValueError("region candidate points must lie within the coded box [-1, 1]")
    return region


def _validate_fixed_runs(
    fixed_runs: np.ndarray | None, *, n_factors: int, n_runs: int
) -> np.ndarray:
    if fixed_runs is None:
        return np.empty((0, n_factors), dtype=float)

    fixed = np.asarray(fixed_runs, dtype=float)
    if fixed.ndim != 2:
        raise ValueError("fixed_runs must be a 2-D array with shape (n_fixed, n_factors)")
    if fixed.shape[1] != n_factors:
        raise ValueError(
            f"fixed_runs has {fixed.shape[1]} columns but factors has {n_factors} entries"
        )
    if fixed.shape[0] > n_runs:
        raise ValueError("fixed_runs cannot contain more rows than n_runs")
    if not np.all(np.isfinite(fixed)):
        raise ValueError("fixed_runs must contain only finite coded values")
    return fixed


def _expand_design_points(
    coded: np.ndarray,
    *,
    factors: FactorSet,
    region: np.ndarray,
    order: int,
    interactions: bool,
) -> np.ndarray:
    expanded = expand_coded_points(
        np.vstack([coded, region]), factors, order=order, interactions=interactions
    )
    return expanded.X[: coded.shape[0]]


def _score_d_optimal(
    coded: np.ndarray,
    *,
    factors: FactorSet,
    region: np.ndarray,
    order: int,
    interactions: bool,
) -> float:
    x = _expand_design_points(
        coded, factors=factors, region=region, order=order, interactions=interactions
    )
    return log_det_information(x)


def _score_i_optimal(
    coded: np.ndarray,
    *,
    factors: FactorSet,
    region: np.ndarray,
    order: int,
    interactions: bool,
) -> float:
    expanded = expand_coded_points(
        np.vstack([coded, region]), factors, order=order, interactions=interactions
    )
    x, f_region = expanded.X[: coded.shape[0]], expanded.X[coded.shape[0] :]
    info = x.T @ x
    sign, _ = np.linalg.slogdet(info)
    if sign <= 0:
        return float("inf")
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return float("inf")
    variances = np.einsum("ij,jk,ik->i", f_region, info_inv, f_region)
    return float(np.mean(variances))


def _objective_score(
    coded: np.ndarray,
    *,
    criterion: Criterion,
    factors: FactorSet,
    region: np.ndarray,
    order: int,
    interactions: bool,
) -> float:
    if criterion == "D":
        return _score_d_optimal(
            coded, factors=factors, region=region, order=order, interactions=interactions
        )
    if criterion == "I":
        score = _score_i_optimal(
            coded, factors=factors, region=region, order=order, interactions=interactions
        )
        return -score if np.isfinite(score) else float("-inf")
    raise ValueError("criterion must be 'D' or 'I'")


def _decode_design(factors: FactorSet, coded: np.ndarray) -> pd.DataFrame:
    columns: dict[str, np.ndarray | list[object]] = {}
    for j, factor in enumerate(factors):
        if isinstance(factor, ContinuousFactor):
            columns[factor.name] = factor.decode(coded[:, j])
        else:
            levels = np.linspace(-1.0, 1.0, len(factor.levels))
            nearest = np.abs(coded[:, [j]] - levels[None, :]).argmin(axis=1)
            valid = np.isclose(coded[:, j], levels[nearest], atol=1e-9, rtol=0.0)
            if not np.all(valid):
                bad = np.unique(coded[~valid, j])
                raise ValueError(
                    f"factor {factor.name!r} categorical candidate coordinate(s) "
                    f"{bad.tolist()} do not match discrete coded levels {levels.tolist()}"
                )
            columns[factor.name] = [factor.levels[int(i)] for i in nearest]
    return pd.DataFrame(columns, index=np.arange(coded.shape[0]))


def _initial_design(
    rng: np.random.Generator,
    *,
    region: np.ndarray,
    fixed: np.ndarray,
    n_runs: int,
) -> np.ndarray:
    n_mutable = n_runs - fixed.shape[0]
    if n_mutable == 0:
        return fixed.copy()

    replace = n_mutable > region.shape[0]
    idx = rng.choice(region.shape[0], size=n_mutable, replace=replace)
    mutable = region[idx]
    return np.vstack([fixed, mutable]) if fixed.size else mutable.copy()


# --------------------------------------------------------------------------- #
# 2.2 Coordinate-exchange engine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimalDesign:
    """The full report from a coordinate-exchange search.

    ``score`` is ``log|X^T X|`` for ``criterion="D"`` or the average prediction variance for
    ``criterion="I"``. ``design`` is the resulting :class:`~doe.design.Design`; the remaining
    fields describe the search so the result is reproducible and self-describing.
    """

    design: Design
    criterion: str
    score: float
    d_efficiency: float
    n_restarts: int
    converged: bool


def coordinate_exchange(
    factors: Sequence[Factor],
    *,
    n_runs: int,
    model: ModelSpec = "quadratic",
    criterion: Criterion = "D",
    region: np.ndarray | None = None,
    n_restarts: int = 20,
    seed: int | None = None,
    fixed_runs: np.ndarray | None = None,
    max_iter: int = 100,
) -> OptimalDesign:
    """Build an optimal design by coordinate exchange (Meyer-Nachtsheim).

    Seeds an ``n_runs x k`` start of feasible coded points (for augmentation the first
    ``len(fixed_runs)`` rows are ``fixed_runs`` and are never exchanged), then repeatedly tries
    every discrete candidate replacement for each mutable run. Each trial recomputes
    ``log|X^T X|`` from scratch with :func:`numpy.linalg.slogdet` through
    :func:`doe.analysis.diagnostics.log_det_information`; this is intentionally simple and
    correctness-first before adding determinant-update shortcuts. Sweeps iterate to convergence
    (or ``max_iter``); the whole search restarts ``n_restarts`` times from fresh random starts
    and keeps the best, guarding against local optima.

    ``model`` selects the term set (``"linear"``/``"quadratic"``); ``region`` is an explicit
    candidate set (defaults to :func:`candidate_grid`); ``seed`` makes the search reproducible.
    The seed actually used is recorded in ``meta["seed"]`` -- when ``seed`` is ``None`` a
    concrete one is drawn first (as :meth:`doe.Design.randomize` does), so a serialized
    optimal design can always regenerate its search.
    """
    if criterion not in {"D", "I"}:
        raise ValueError("criterion must be 'D' or 'I'")
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    fs = FactorSet(factors)
    n_factors = len(fs)
    region = _validate_region(
        candidate_grid(list(fs)) if region is None else region, n_factors=n_factors
    )
    fixed = _validate_fixed_runs(fixed_runs, n_factors=n_factors, n_runs=n_runs)
    order, interactions = _model_spec(model)

    n_terms = expand_coded_points(region, fs, order=order, interactions=interactions).X.shape[1]
    if n_runs < n_terms:
        raise ValueError(
            f"n_runs={n_runs} cannot estimate the {n_terms}-term {model!r} model"
        )

    # resolve an unset seed to a concrete drawn value (as Design.randomize does) so the
    # search recorded in meta is always regenerable from a serialized design.
    seed = _draw_seed(seed)
    rng = np.random.default_rng(seed)
    first_mutable = fixed.shape[0]
    best_design: np.ndarray | None = None
    best_score = float("-inf")
    best_converged = False
    tol = 1e-12

    for _restart in range(n_restarts):
        current = _initial_design(rng, region=region, fixed=fixed, n_runs=n_runs)
        current_score = _objective_score(
            current,
            criterion=criterion,
            factors=fs,
            region=region,
            order=order,
            interactions=interactions,
        )
        converged = False

        for _iteration in range(max_iter):
            improved = False
            for row in range(first_mutable, n_runs):
                row_best = current[row].copy()
                row_best_score = current_score
                for candidate in region:
                    trial = current.copy()
                    trial[row] = candidate
                    score = _objective_score(
                        trial,
                        criterion=criterion,
                        factors=fs,
                        region=region,
                        order=order,
                        interactions=interactions,
                    )
                    if score > row_best_score + tol:
                        row_best = candidate.copy()
                        row_best_score = score

                if row_best_score > current_score + tol:
                    current[row] = row_best
                    current_score = row_best_score
                    improved = True

            if not improved:
                converged = True
                break

        if current_score > best_score + tol:
            best_design = current.copy()
            best_score = current_score
            best_converged = converged

    if best_design is None:  # pragma: no cover - n_restarts validation makes this unreachable
        raise RuntimeError("coordinate_exchange failed to produce a design")

    d_score = _score_d_optimal(
        best_design, factors=fs, region=region, order=order, interactions=interactions
    )
    report_score = (
        d_score
        if criterion == "D"
        else _score_i_optimal(
            best_design, factors=fs, region=region, order=order, interactions=interactions
        )
    )
    d_efficiency = float(np.exp(d_score / n_terms) / n_runs) if np.isfinite(d_score) else 0.0
    meta: dict[str, object] = {
        "criterion": criterion,
        "score": report_score,
        "d_efficiency": d_efficiency,
        "n_restarts": n_restarts,
        "seed": seed,
        "model": model,
        "order": order,
        "interactions": interactions,
        "converged": best_converged,
    }
    design = Design(
        _decode_design(fs, best_design),
        fs,
        name=f"{criterion.lower()}_optimal_{model}",
        meta=meta,
    )
    return OptimalDesign(
        design=design,
        criterion=criterion,
        score=report_score,
        d_efficiency=d_efficiency,
        n_restarts=n_restarts,
        converged=best_converged,
    )


# --------------------------------------------------------------------------- #
# 2.3 Public generators (intention-revealing wrappers over the engine)
# --------------------------------------------------------------------------- #


def d_optimal(
    factors: Sequence[Factor], *, n_runs: int, model: ModelSpec = "quadratic", **kwargs: Any
) -> Design:
    """A D-optimal design: maximise ``log|X^T X|`` for ``model`` over ``n_runs`` runs.

    Thin wrapper over :func:`coordinate_exchange` with ``criterion="D"``; extra keyword
    arguments (``region``, ``n_restarts``, ``seed``, ``max_iter``) are forwarded, so e.g.
    ``d_optimal(factors, n_runs=12, seed=0)`` is reproducible. Returns a plain
    :class:`~doe.design.Design` with the search diagnostics (criterion, score, d_efficiency,
    n_restarts, seed) carried in ``design.meta``.
    """
    return coordinate_exchange(
        factors, n_runs=n_runs, model=model, criterion="D", **kwargs
    ).design


def i_optimal(
    factors: Sequence[Factor], *, n_runs: int, model: ModelSpec = "quadratic", **kwargs: Any
) -> Design:
    """An I-optimal design: minimise the average prediction variance over the region.

    Thin wrapper over :func:`coordinate_exchange` with ``criterion="I"`` (same forwarded
    keywords as :func:`d_optimal`). Returns a plain :class:`~doe.design.Design` with diagnostics
    in ``design.meta``. For the same model/budget it has lower average prediction variance than
    the D-optimal design (the criteria genuinely differ).
    """
    return coordinate_exchange(
        factors, n_runs=n_runs, model=model, criterion="I", **kwargs
    ).design


def augment(
    design: Design,
    *,
    n_runs: int,
    model: ModelSpec = "quadratic",
    criterion: Criterion = "D",
    **kwargs: Any,
) -> Design:
    """Augment ``design`` with ``n_runs`` extra optimal runs (same engine, seeded fixed rows).

    Runs :func:`coordinate_exchange` with ``fixed_runs`` set to ``design``'s coded rows (other
    keywords forwarded). The existing rows are held byte-for-byte (tagged
    ``point_type="existing"``) and only ``n_runs`` new rows are searched (tagged ``"augment"``).
    The augmented ``|X^T X|`` is never smaller than the original -- extra optimal runs cannot
    reduce information.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if "fixed_runs" in kwargs:
        raise TypeError("augment manages fixed_runs from the existing design")

    fixed = coded_design_points(design)
    result = coordinate_exchange(
        list(design.factors),
        n_runs=design.n_runs + n_runs,
        model=model,
        criterion=criterion,
        fixed_runs=fixed,
        **kwargs,
    ).design
    return Design(
        result.runs,
        result.factors,
        name=f"augmented_{result.name}",
        meta={**result.meta, "n_existing": design.n_runs, "n_augmented": n_runs},
        point_types=("existing",) * design.n_runs + ("augment",) * n_runs,
    )
