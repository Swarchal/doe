"""Space-filling design generators.

Designs that target *coverage* of the region rather than efficient estimation of a fixed
polynomial model -- for computer experiments, surrogate modelling, and exploratory
sampling of expensive simulations:

    * :func:`latin_hypercube` -- stratified sampling, optionally maximin/correlation-optimized
    * :func:`sobol`           -- scrambled Sobol' low-discrepancy sequence (powers of 2 only)
    * :func:`halton`          -- scrambled Halton low-discrepancy sequence (any run count)

All three are thin wrappers over ``scipy.stats.qmc``: samples in ``[0, 1]^k`` are mapped to
coded ``[-1, +1]`` and decoded to natural units, so the returned :class:`~doe.design.Design`
flows through the rest of the library unchanged. Continuous factors only. Coverage is judged
by :func:`doe.analysis.diagnostics.discrepancy` and
:func:`doe.analysis.diagnostics.maximin_distance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from ..design import Design, _draw_seed
from ..factors import ContinuousFactor, Factor, FactorSet

#: Post-sampling optimization applied to the Latin hypercube (``None`` = plain LHS).
LhsCriterion = Literal["maximin", "correlation"]

#: Candidate hypercubes drawn per call when optimizing for ``criterion``.
_N_CANDIDATES = 30


def _validate_continuous(factors: Sequence[Factor]) -> list[ContinuousFactor]:
    """Reject anything but continuous factors -- space-filling has no categorical coding."""
    non_continuous = [f.name for f in factors if not isinstance(f, ContinuousFactor)]
    if non_continuous:
        raise ValueError(
            "space-filling designs support continuous factors only; got non-continuous "
            f"factors: {non_continuous}"
        )
    return [f for f in factors if isinstance(f, ContinuousFactor)]


def _decode_unit_sample(sample: np.ndarray, factors: Sequence[ContinuousFactor]) -> pd.DataFrame:
    """Map a ``[0, 1]^k`` qmc sample to coded ``[-1, +1]`` then to natural units."""
    coded = 2.0 * sample - 1.0
    data = {factor.name: factor.decode(coded[:, j]) for j, factor in enumerate(factors)}
    return pd.DataFrame(data)


def _best_by_criterion(candidates: list[np.ndarray], criterion: LhsCriterion) -> np.ndarray:
    """Pick the candidate hypercube that best satisfies ``criterion``."""
    if criterion == "maximin":
        scores = [pdist(c).min() for c in candidates]
        return candidates[int(np.argmax(scores))]
    if criterion == "correlation":
        # the criterion minimises the largest *pairwise* column correlation, of which a
        # one-factor design has none: np.corrcoef would hand back a 0-d array and
        # np.fill_diagonal would fail on it with a bare "array must be at least 2-d"
        if candidates[0].shape[1] < 2:
            raise ValueError(
                "criterion='correlation' needs at least 2 factors (it minimises the largest "
                "correlation *between* columns); use criterion='maximin' or None"
            )
        scores = []
        for c in candidates:
            corr = np.corrcoef(c, rowvar=False)
            np.fill_diagonal(corr, 0.0)
            scores.append(np.abs(corr).max())
        return candidates[int(np.argmin(scores))]
    raise ValueError(f"unknown criterion {criterion!r}; expected 'maximin' or 'correlation'")


def latin_hypercube(
    factors: Sequence[Factor],
    *,
    n_runs: int,
    criterion: LhsCriterion | None = "maximin",
    seed: int | None = None,
) -> Design:
    """Generate a Latin hypercube design over continuous factors.

    Each factor's coded range is split into ``n_runs`` equal strata with exactly one run
    per stratum -- guaranteed one-dimensional uniformity for any run count. When
    ``criterion`` is set, ``_N_CANDIDATES`` independent hypercubes are drawn (each still a
    valid stratified sample) and the one scoring best on the criterion is kept.

    Args:
        factors: the continuous factors to vary.
        criterion: post-optimization of the hypercube -- ``"maximin"`` (default) spreads
            points apart, ``"correlation"`` minimizes pairwise column correlation,
            ``None`` takes the raw stratified sample.
        seed: RNG seed, recorded in ``meta["seed"]`` for reproducibility.

    Returns:
        A :class:`Design` in natural units with ``meta`` recording
        ``{"sampler": "lhs", "criterion": ..., "seed": ...}``.
    """
    continuous = _validate_continuous(factors)
    resolved_seed = _draw_seed(seed)
    k = len(continuous)

    if criterion is None:
        sampler = qmc.LatinHypercube(d=k, scramble=True, seed=resolved_seed)
        sample = sampler.random(n_runs)
    else:
        rng = np.random.default_rng(resolved_seed)
        candidates = []
        for _ in range(_N_CANDIDATES):
            candidate_seed = int(rng.integers(0, 2**32 - 1))
            sampler = qmc.LatinHypercube(d=k, scramble=True, seed=candidate_seed)
            candidates.append(sampler.random(n_runs))
        sample = _best_by_criterion(candidates, criterion)

    runs = _decode_unit_sample(sample, continuous)
    meta: dict[str, object] = {"sampler": "lhs", "criterion": criterion, "seed": resolved_seed}
    return Design(runs, FactorSet(continuous), meta=meta)


def sobol(
    factors: Sequence[Factor],
    *,
    n_runs: int,
    scramble: bool = True,
    seed: int | None = None,
) -> Design:
    """Generate a Sobol' low-discrepancy design over continuous factors.

    Sobol' points only achieve their balance guarantees in blocks of ``2^m``, so
    ``n_runs`` must be a power of two (a ``ValueError`` names the nearest valid sizes);
    use :func:`halton` or :func:`latin_hypercube` for arbitrary run counts.

    Args:
        factors: the continuous factors to vary.
        scramble: Owen-scramble the sequence (default ``True``; also what makes ``seed``
            meaningful).
        seed: RNG seed for scrambling, recorded in ``meta["seed"]``.

    Returns:
        A :class:`Design` in natural units with ``meta`` recording
        ``{"sampler": "sobol", "scramble": ..., "seed": ...}``.
    """
    continuous = _validate_continuous(factors)
    if n_runs <= 0 or (n_runs & (n_runs - 1)) != 0:
        lower = 1 << (n_runs.bit_length() - 1) if n_runs > 0 else 1
        upper = lower if lower == n_runs else lower * 2
        raise ValueError(
            f"sobol requires a power-of-two n_runs; got {n_runs} "
            f"(nearest valid sizes: {lower}, {upper})"
        )
    resolved_seed = _draw_seed(seed)
    sampler = qmc.Sobol(d=len(continuous), scramble=scramble, seed=resolved_seed)
    sample = sampler.random(n_runs)
    runs = _decode_unit_sample(sample, continuous)
    meta: dict[str, object] = {"sampler": "sobol", "scramble": scramble, "seed": resolved_seed}
    return Design(runs, FactorSet(continuous), meta=meta)


def halton(
    factors: Sequence[Factor],
    *,
    n_runs: int,
    scramble: bool = True,
    seed: int | None = None,
) -> Design:
    """Generate a Halton low-discrepancy design over continuous factors.

    Lower-quality coverage than :func:`sobol` in high dimension, but valid for any
    ``n_runs`` -- the escape hatch when the budget is not a power of two.

    Args:
        factors: the continuous factors to vary.
        scramble: scramble the sequence (default ``True``).
        seed: RNG seed for scrambling, recorded in ``meta["seed"]``.

    Returns:
        A :class:`Design` in natural units with ``meta`` recording
        ``{"sampler": "halton", "scramble": ..., "seed": ...}``.
    """
    continuous = _validate_continuous(factors)
    resolved_seed = _draw_seed(seed)
    sampler = qmc.Halton(d=len(continuous), scramble=scramble, seed=resolved_seed)
    sample = sampler.random(n_runs)
    runs = _decode_unit_sample(sample, continuous)
    meta: dict[str, object] = {"sampler": "halton", "scramble": scramble, "seed": resolved_seed}
    return Design(runs, FactorSet(continuous), meta=meta)
