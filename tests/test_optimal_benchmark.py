"""Benchmark + equivalence guards for the coordinate-exchange fast path.

The engine scores each candidate swap from a model matrix that is expanded *once* (region +
fixed rows up front), substituting a single already-expanded row per trial, instead of
re-expanding the whole candidate region on every trial as the original loop did. Two things
must hold, and these tests pin both:

1. **Exactness** -- the fast per-trial scorer (:func:`_fast_objective`) returns the same value
   as scoring the design from scratch (:func:`_score_d_optimal` / :func:`_score_i_optimal`,
   which re-expand ``vstack([coded, region])`` exactly as the old inner loop did). The
   optimization must not change a single reported number.
2. **Speed** -- the fast scorer is materially faster than the from-scratch scorer, so a
   regression that reintroduces per-trial expansion is caught. The measured gain is ~5-12x;
   the threshold here is deliberately loose (>= 2.5x) to stay robust under CI noise.
"""

import time

import numpy as np
import pytest

from doe.analysis.model import expand_coded_points
from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet
from doe.generators.optimal import (
    _d_row_logdets,
    _d_sweep_state,
    _fast_objective,
    _score_d_optimal,
    _score_i_optimal,
    candidate_grid,
    coordinate_exchange,
)


def _cont(n):
    return [ContinuousFactor(chr(ord("a") + i), low=-1.0, high=1.0) for i in range(n)]


def _setup(factors, *, n_runs, order, interactions, seed):
    """A random design drawn from the candidate region, plus the precomputed region matrix."""
    fs = FactorSet(factors)
    region = candidate_grid(list(fs))
    f_region = expand_coded_points(region, fs, order=order, interactions=interactions).X
    rng = np.random.default_rng(seed)
    idx = rng.choice(region.shape[0], size=n_runs, replace=n_runs > region.shape[0])
    coded = region[idx]
    f_design = f_region[idx]
    return fs, region, f_region, coded, f_design


# --------------------------------------------------------------------------- #
# Exactness: the fast scorer equals scoring from scratch, term for term
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_fast_objective_matches_from_scratch_scoring(criterion):
    fs, region, f_region, _coded, _f_design = _setup(
        _cont(3), n_runs=15, order=2, interactions=True, seed=1
    )
    score_fn = _fast_objective(criterion, f_region)

    for _ in range(50):  # many random designs, including some singular starts
        rng_idx = np.random.default_rng(_).choice(region.shape[0], size=15)
        design = region[rng_idx]
        f_des = f_region[rng_idx]

        fast = score_fn(f_des)
        if criterion == "D":
            slow = _score_d_optimal(
                design, factors=fs, region=region, order=2, interactions=True
            )
        else:
            raw = _score_i_optimal(
                design, factors=fs, region=region, order=2, interactions=True
            )
            slow = -raw if np.isfinite(raw) else float("-inf")

        if np.isinf(fast) or np.isinf(slow):
            assert np.isinf(fast) and np.isinf(slow)  # both flag the singular design
        else:
            assert fast == pytest.approx(slow, rel=1e-12, abs=1e-12)


def test_fast_objective_matches_from_scratch_with_categorical():
    factors = [
        ContinuousFactor("temp", -1.0, 1.0),
        ContinuousFactor("time", -1.0, 1.0),
        CategoricalFactor("cat", ("A", "B", "C")),
    ]
    fs, region, f_region, coded, f_design = _setup(
        factors, n_runs=12, order=2, interactions=True, seed=2
    )
    score_fn = _fast_objective("D", f_region)

    fast = score_fn(f_design)
    slow = _score_d_optimal(coded, factors=fs, region=region, order=2, interactions=True)
    assert fast == pytest.approx(slow, rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- #
# Rank-1 determinant update: identical to scoring every candidate from scratch
# --------------------------------------------------------------------------- #


def test_row_logdets_d_matches_per_candidate_slogdet():
    # The rank-1 update (shared per-sweep state + per-row scoring) must return, for every
    # candidate, exactly the log|X^T X| a from-scratch slogdet of the swapped design would give.
    _fs, region, f_region, _coded, f_design = _setup(
        _cont(3), n_runs=15, order=2, interactions=True, seed=7
    )
    info = f_design.T @ f_design
    row = 4

    state = _d_sweep_state(info, f_region)
    assert state is not None  # non-saturated design -> M is comfortably positive definite
    logdets = _d_row_logdets(state, f_design[row])
    assert logdets is not None  # a low-leverage run -> 1 - d_a is well away from 0

    for cand in range(region.shape[0]):
        trial = f_design.copy()
        trial[row] = f_region[cand]
        sign, expected = np.linalg.slogdet(trial.T @ trial)
        assert sign > 0
        assert logdets[cand] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_row_logdets_d_falls_back_on_rank_deficient_partial_design():
    # A saturated design (n_runs == n_terms) loses rank when any run is removed, so 1 - d_a
    # collapses (unit leverage) and the rank-1 update is undefined -- the per-row scorer must
    # return None so the engine drops to its exact per-candidate path (rather than lying). If the
    # random start is itself singular, the shared state is None, which triggers the same fallback.
    _fs, _region, f_region, _coded, f_design = _setup(
        _cont(2), n_runs=6, order=2, interactions=True, seed=8
    )  # k=2 quadratic -> 6 terms == 6 runs
    info = f_design.T @ f_design
    state = _d_sweep_state(info, f_region)
    assert state is None or _d_row_logdets(state, f_design[0]) is None


# --------------------------------------------------------------------------- #
# End-to-end: the reported score still equals a from-scratch refit of the result
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_reported_score_equals_full_refit(criterion):
    factors = _cont(3)
    result = coordinate_exchange(
        factors, n_runs=15, model="quadratic", criterion=criterion, seed=0, n_restarts=5
    )
    coded = result.design.coded().to_numpy(dtype=float)
    if criterion == "D":
        expected = _score_d_optimal(
            coded, factors=result.design.factors, region=candidate_grid(factors),
            order=2, interactions=True,
        )
    else:
        expected = _score_i_optimal(
            coded, factors=result.design.factors, region=candidate_grid(factors),
            order=2, interactions=True,
        )
    assert result.score == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Speed: the fast scorer beats from-scratch scoring by a wide margin
# --------------------------------------------------------------------------- #


def test_fast_objective_is_faster_than_from_scratch():
    fs, region, f_region, coded, f_design = _setup(
        _cont(4), n_runs=18, order=2, interactions=True, seed=3
    )
    score_fn = _fast_objective("D", f_region)
    n_evals = 300

    score_fn(f_design)  # warm up
    _score_d_optimal(coded, factors=fs, region=region, order=2, interactions=True)

    t0 = time.perf_counter()
    for _ in range(n_evals):
        score_fn(f_design)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_evals):
        _score_d_optimal(coded, factors=fs, region=region, order=2, interactions=True)
    t_slow = time.perf_counter() - t0

    speedup = t_slow / t_fast
    assert speedup >= 2.5, f"fast scorer only {speedup:.1f}x faster (t_fast={t_fast:.3f}s)"
