"""Optimal (computer-generated) designs via coordinate exchange.

Where the named recipes (factorial, CCD, Box-Behnken) produce a fixed run set for a fixed
model, the coordinate-exchange engine here *builds* a run set to maximise a chosen criterion
(D- or I-optimality) for a user-specified model, run budget, and candidate region. This is
what handles irregular constraints, custom models, odd run counts, and augmenting an existing
design -- the cases the recipes can't.

The engine's objective is a diagnostic: ``criterion="D"`` maximises
:func:`~doe.analysis.diagnostics.log_det_information`; ``criterion="I"`` minimises the average
prediction variance over the region. Generators return a plain :class:`~doe.design.Design`
(so ``coded()``/fitting/ANOVA/plots work unchanged) with the search diagnostics stashed in
``design.meta``.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..analysis.diagnostics import log_det_information
from ..analysis.model import coded_design_points, expand_coded_points
from ..design import Design, _draw_seed
from ..factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet, MixtureFactor

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
        elif isinstance(factor, MixtureFactor):
            raise TypeError(
                "candidate_grid covers the coded box, not the mixture simplex; use "
                "doe.generators.mixture.mixture_candidates for mixture components"
            )
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


def _decode_design(factors: FactorSet, coded: np.ndarray) -> pd.DataFrame:
    columns: dict[str, np.ndarray | list[object]] = {}
    for j, factor in enumerate(factors):
        if isinstance(factor, ContinuousFactor):
            columns[factor.name] = factor.decode(coded[:, j])
        elif isinstance(factor, MixtureFactor):
            columns[factor.name] = coded[:, j]  # proportions are already natural units
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


def _fast_objective(
    criterion: Criterion, f_region: np.ndarray
) -> Callable[[np.ndarray], float]:
    """A closure scoring a *precomputed* design model matrix -- the hot-loop objective.

    The coordinate-exchange sweeps score one candidate replacement at a time, so the model
    matrix is assembled once (region + fixed rows expanded up front) and each trial only
    substitutes a single already-expanded row. This closure evaluates the criterion straight
    from that ``n_runs x n_terms`` block, skipping the per-trial re-expansion of the whole
    candidate region that dominated the old inner loop. It is numerically identical to scoring
    the design from scratch: D returns ``log|X^T X|`` (``-inf`` if singular); I returns the
    negated average prediction variance over ``f_region`` (``-inf`` if singular), so both are
    maximised.
    """
    if criterion == "D":

        def score_d(f_design: np.ndarray) -> float:
            sign, logdet = np.linalg.slogdet(f_design.T @ f_design)
            return float(logdet) if sign > 0 else float("-inf")

        return score_d

    def score_i(f_design: np.ndarray) -> float:
        info = f_design.T @ f_design
        sign, _ = np.linalg.slogdet(info)
        if sign <= 0:
            return float("-inf")
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return float("-inf")
        variances = np.einsum("ij,jk,ik->i", f_region, info_inv, f_region)
        return -float(np.mean(variances))

    return score_i


def _row_logdets_d(info: np.ndarray, a: np.ndarray, f_region: np.ndarray) -> np.ndarray | None:
    """``log|X^T X|`` for *every* candidate in one shot, via the rank-1 determinant lemma.

    A coordinate exchange at one run swaps that run's model vector out (``a``) and a candidate
    ``b`` in, so the information matrix goes ``M -> M - aa^T + bb^T``. Writing
    ``M_minus = M - aa^T`` (the design *without* the swept run -- a Gram matrix, hence positive
    semidefinite) the matrix determinant lemma gives a closed form that shares ``M_minus`` across
    all candidates::

        log|M_minus + bb^T| = log|M_minus| + log(1 + b^T M_minus^{-1} b).

    So one Cholesky + one inverse of ``M_minus`` (both ``O(p^3)``, done once per run) plus a
    single vectorised quadratic form over ``f_region`` replaces the ``n_candidates`` separate
    ``slogdet`` calls the naive sweep makes. Because ``M_minus`` is PD its inverse is PD, so
    ``b^T M_minus^{-1} b >= 0`` and every returned value is finite and real.

    Returns the ``(n_candidates,)`` vector of full ``log|X^T X|`` scores, or ``None`` when
    ``M_minus`` is not positive definite (a rank-deficient partial design -- common on random
    starts); the caller then falls back to the exact per-candidate ``slogdet`` path. The result
    is the D-optimality objective, mathematically identical to scoring each candidate from
    scratch.
    """
    m_minus = info - np.outer(a, a)
    try:
        chol = np.linalg.cholesky(m_minus)
    except np.linalg.LinAlgError:
        return None
    # Cholesky can slip past a *numerically* rank-deficient M_minus (e.g. a saturated design
    # that loses rank when a run is removed) with a tiny positive pivot, yielding a garbage
    # inverse. Gate on the pivot spread: a ratio this small means condition number > ~1e16,
    # i.e. effectively singular -> None -> the caller's exact per-candidate fallback, which
    # also explores such rows more thoroughly than a rank-1 shortcut could.
    diag = np.diagonal(chol)
    if float(diag.min()) <= 1e-6 * float(diag.max()):
        return None
    g = np.linalg.inv(m_minus)
    logdet_minus = 2.0 * float(np.sum(np.log(diag)))
    quad = np.einsum("ij,jk,ik->i", f_region, g, f_region)
    return np.asarray(logdet_minus + np.log1p(quad), dtype=float)


# --------------------------------------------------------------------------- #
# 2.2 Coordinate-exchange engine
# --------------------------------------------------------------------------- #


def _run_restart(
    rng: np.random.Generator,
    *,
    region: np.ndarray,
    f_region: np.ndarray,
    f_fixed: np.ndarray,
    fixed: np.ndarray,
    criterion: Criterion,
    n_runs: int,
    first_mutable: int,
    n_mutable: int,
    max_iter: int,
    tol: float,
) -> tuple[float, np.ndarray, bool]:
    """One coordinate-exchange restart: random start, sweep to convergence, return the result.

    Pure in ``rng`` -- the only source of randomness is the initial draw of mutable rows, so a
    restart is fully determined by the generator it is handed. Returns
    ``(score, coded_design, converged)``; the caller keeps the best across restarts. Factored out
    of :func:`coordinate_exchange` so restarts can run independently, in-process (sequential) or
    across worker processes (``n_jobs``).
    """
    score_fn = _fast_objective(criterion, f_region)

    # Draw the mutable rows as region indices, then carry the coded points and their model rows
    # together (the model rows are the once-expanded region, indexed -- never re-expanded).
    current = np.empty((n_runs, region.shape[1]), dtype=float)
    f_current = np.empty((n_runs, f_region.shape[1]), dtype=float)
    if first_mutable:
        current[:first_mutable] = fixed
        f_current[:first_mutable] = f_fixed
    if n_mutable:
        replace = n_mutable > region.shape[0]
        idx = rng.choice(region.shape[0], size=n_mutable, replace=replace)
        current[first_mutable:] = region[idx]
        f_current[first_mutable:] = f_region[idx]

    current_score = score_fn(f_current)
    # D-optimality scores every candidate for a run with one rank-1 determinant update (see
    # _row_logdets_d), so it keeps the running information matrix M = X^T X. Other criteria (I)
    # score from scratch per candidate and don't need M.
    info = f_current.T @ f_current if criterion == "D" else None
    converged = False

    for _iteration in range(max_iter):
        improved = False
        for row in range(first_mutable, n_runs):
            logdets = (
                _row_logdets_d(info, f_current[row], f_region) if info is not None else None
            )
            if logdets is not None:
                # Vectorised rank-1 path: pick the best candidate from one score vector.
                best_cand = int(np.argmax(logdets))
                row_best_score = float(logdets[best_cand])
                if row_best_score > current_score + tol:
                    f_current[row] = f_region[best_cand]
                    current[row] = region[best_cand]
                    info = f_current.T @ f_current  # refresh exactly (no drift)
                    current_score = float(np.linalg.slogdet(info)[1])
                    improved = True
                continue

            # Fallback: exact per-candidate scoring (I-optimality, or a rank-deficient partial
            # design where the rank-1 update is undefined).
            row_orig = f_current[row].copy()
            best_cand = -1
            row_best_score = current_score
            for cand in range(region.shape[0]):
                f_current[row] = f_region[cand]
                score = score_fn(f_current)
                if score > row_best_score + tol:
                    best_cand = cand
                    row_best_score = score

            if best_cand >= 0 and row_best_score > current_score + tol:
                f_current[row] = f_region[best_cand]
                current[row] = region[best_cand]
                if info is not None:
                    info = f_current.T @ f_current
                current_score = row_best_score
                improved = True
            else:
                f_current[row] = row_orig  # no improvement: restore the original row

        if not improved:
            converged = True
            break

    return current_score, current, converged


# Worker-process plumbing for parallel restarts. The read-only arrays/scalars a restart needs
# are shipped to each worker exactly once via the pool initializer (not re-pickled per task);
# each task then only carries a spawned seed.
_RESTART_CTX: dict[str, Any] = {}


def _restart_pool_init(ctx: dict[str, Any]) -> None:
    _RESTART_CTX.clear()
    _RESTART_CTX.update(ctx)


def _restart_pool_task(seed_seq: np.random.SeedSequence) -> tuple[float, np.ndarray, bool]:
    return _run_restart(np.random.default_rng(seed_seq), **_RESTART_CTX)


def _resolve_workers(n_jobs: int, *, n_restarts: int) -> int:
    """Number of worker processes for ``n_jobs`` (``-1`` == all cores), capped at ``n_restarts``."""
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer, or -1 for all cores")
    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    return max(1, min(workers, n_restarts))


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
    n_jobs: int = 1,
) -> OptimalDesign:
    """Build an optimal design by coordinate exchange (Meyer-Nachtsheim).

    Seeds an ``n_runs x k`` start of feasible coded points (for augmentation the first
    ``len(fixed_runs)`` rows are ``fixed_runs`` and are never exchanged), then repeatedly tries
    every discrete candidate replacement for each mutable run. Two layers of speedup keep the
    result numerically identical to a from-scratch refit: (1) the candidate region (and any
    fixed rows) is expanded into model rows *once*, up front, so a trial only substitutes a
    single already-expanded row; (2) for ``criterion="D"`` all candidates for a run are scored
    together by a rank-1 determinant update (:func:`_row_logdets_d`) -- one Cholesky + inverse
    of the run-deleted information matrix instead of an :func:`numpy.linalg.slogdet` per
    candidate -- with an exact per-candidate fallback whenever that partial design is
    rank-deficient. Sweeps iterate to convergence (or ``max_iter``); the whole search restarts
    ``n_restarts`` times from fresh random starts and keeps the best, guarding against local
    optima.

    ``model`` selects the term set (``"linear"``/``"quadratic"``); ``region`` is an explicit
    candidate set (defaults to :func:`candidate_grid`, or to
    :func:`~doe.generators.mixture.mixture_candidates` for an all-mixture factor set -- whose
    points are proportions, expanded through the Scheffé model, and exchanged whole so
    sum-to-1 holds by construction); ``seed`` makes the search reproducible.
    The seed actually used is recorded in ``meta["seed"]`` -- when ``seed`` is ``None`` a
    concrete one is drawn first (as :meth:`doe.Design.randomize` does), so a serialized
    optimal design can always regenerate its search.

    ``n_jobs`` runs the (independent) restarts in parallel across worker processes -- ``1``
    (default) stays single-process, ``-1`` uses all cores, ``k`` uses ``k`` (capped at
    ``n_restarts``). This is an opt-in speedup for *large* searches (many factors/candidates,
    high ``n_restarts``); for quick searches the process start-up and pickling overhead outweighs
    the gain, so leave it at ``1``. The parallel path seeds each restart from an independent child
    stream, so its result depends only on ``(seed, n_restarts)`` and is identical whatever
    ``n_jobs > 1`` you pick -- but it is *not* the same design the single-process default produces
    for that seed (the two seed their restarts differently, both fully reproducible). Prefer
    running one large ``n_jobs`` search over many concurrent single-job calls, and note the
    ``doe-service`` HTTP endpoints deliberately do not expose ``n_jobs`` (a request must not fan
    out across the server's cores).
    """
    if criterion not in {"D", "I"}:
        raise ValueError("criterion must be 'D' or 'I'")
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer, or -1 for all cores")

    fs = FactorSet(factors)
    n_factors = len(fs)
    if region is None:
        if fs.is_mixture:
            # the default box grid is infeasible on the simplex; use the mixture lattice
            from .mixture import mixture_candidates

            region = mixture_candidates(list(fs))
        else:
            region = candidate_grid(list(fs))
    region = _validate_region(region, n_factors=n_factors)
    fixed = _validate_fixed_runs(fixed_runs, n_factors=n_factors, n_runs=n_runs)
    order, interactions = _model_spec(model)

    n_terms = expand_coded_points(region, fs, order=order, interactions=interactions).X.shape[1]
    if n_runs < n_terms:
        raise ValueError(f"n_runs={n_runs} cannot estimate the {n_terms}-term {model!r} model")

    # Expand the candidate region (and any fixed rows) into model rows ONCE, up front. Every
    # design row the search ever holds is either a fixed row or a region candidate, so a trial
    # only ever substitutes one already-expanded row -- no per-trial re-expansion of the whole
    # region (the old inner loop's dominant cost). Fixed rows are stacked with the region before
    # expansion so the squared-term heuristic sees the same value union the old code did (design
    # rows are drawn from the region, so their value set is a subset), keeping the term layout --
    # and therefore the scores -- identical.
    stacked = np.vstack([fixed, region]) if fixed.size else region
    f_all = expand_coded_points(stacked, fs, order=order, interactions=interactions).X
    f_fixed = f_all[: fixed.shape[0]]
    f_region = f_all[fixed.shape[0] :]

    # resolve an unset seed to a concrete drawn value (as Design.randomize does) so the
    # search recorded in meta is always regenerable from a serialized design.
    seed = _draw_seed(seed)
    first_mutable = fixed.shape[0]
    n_mutable = n_runs - first_mutable
    tol = 1e-12
    restart_kwargs: dict[str, Any] = {
        "region": region,
        "f_region": f_region,
        "f_fixed": f_fixed,
        "fixed": fixed,
        "criterion": criterion,
        "n_runs": n_runs,
        "first_mutable": first_mutable,
        "n_mutable": n_mutable,
        "max_iter": max_iter,
        "tol": tol,
    }

    if n_jobs == 1:
        # Sequential: one shared generator drawn restart-by-restart (bit-identical to the
        # single-process engine). This is the default and the reproducibility contract.
        rng = np.random.default_rng(seed)
        results = [_run_restart(rng, **restart_kwargs) for _ in range(n_restarts)]
    else:
        # Parallel restarts. Each restart is seeded by an independent child stream spawned from
        # the master seed, so the outcome depends only on (seed, n_restarts) -- never on the
        # worker count or completion order. (This seeding differs from the sequential path, so a
        # given seed maps to a different -- but equally valid and fully reproducible -- design.)
        child_seeds = np.random.SeedSequence(seed).spawn(n_restarts)
        workers = _resolve_workers(n_jobs, n_restarts=n_restarts)
        if workers == 1:
            results = [
                _run_restart(np.random.default_rng(s), **restart_kwargs) for s in child_seeds
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_restart_pool_init,
                initargs=(restart_kwargs,),
            ) as executor:
                results = list(executor.map(_restart_pool_task, child_seeds))

    # Keep the best restart, preferring the earliest on ties (matches the sequential tie-break
    # and is independent of the order restarts actually finished in).
    best_design: np.ndarray | None = None
    best_score = float("-inf")
    best_converged = False
    for current_score, current, converged in results:
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
    return coordinate_exchange(factors, n_runs=n_runs, model=model, criterion="D", **kwargs).design


def i_optimal(
    factors: Sequence[Factor], *, n_runs: int, model: ModelSpec = "quadratic", **kwargs: Any
) -> Design:
    """An I-optimal design: minimise the average prediction variance over the region.

    Thin wrapper over :func:`coordinate_exchange` with ``criterion="I"`` (same forwarded
    keywords as :func:`d_optimal`). Returns a plain :class:`~doe.design.Design` with diagnostics
    in ``design.meta``. For the same model/budget it has lower average prediction variance than
    the D-optimal design (the criteria genuinely differ).
    """
    return coordinate_exchange(factors, n_runs=n_runs, model=model, criterion="I", **kwargs).design


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
