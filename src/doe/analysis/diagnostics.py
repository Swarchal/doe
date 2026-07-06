"""Design diagnostics (Phase 3a) -- evaluate *any* design against a model.

Phases 1-2 produced *named* designs (factorial, Plackett-Burman, CCD, Box-Behnken).
These functions judge a design -- however it was made -- against a model: the information
matrix and its scalars, variance inflation, leverage, the alias/correlation matrix, and
D/A/G/I-efficiency relative to an orthogonal reference.

Everything here is headless (``slogdet``/``inv``/SVD/QR on the existing coded model
matrix). The cores feed both the plotting wrappers and the coordinate-exchange engine in
:mod:`doe.generators.optimal` (whose D-objective *is* :func:`log_det_information`). See
``docs/PHASE3.md`` for the build plan.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from ..design import Design
from ..factors import CategoricalFactor, ContinuousFactor, FactorSet
from .model import coded_design_points, expand_coded_points

# --------------------------------------------------------------------------- #
# 1.1 Information-matrix scalars
# --------------------------------------------------------------------------- #


def information_matrix(x: np.ndarray) -> np.ndarray:
    """The information matrix ``X^T X`` of a model matrix ``X``."""
    x = np.asarray(x, dtype=float)
    return np.asarray(x.T @ x, dtype=float)


def condition_number(x: np.ndarray) -> float:
    """Condition number of ``X`` (ratio of largest to smallest singular value, via SVD).

    A condition number of ``1`` means perfectly orthogonal columns; large values flag
    near-collinear terms whose coefficients are poorly determined.
    """
    return float(np.linalg.cond(np.asarray(x, dtype=float)))


def log_det_information(x: np.ndarray) -> float:
    """``log|X^T X|`` via ``slogdet`` -- the D-optimality objective.

    Uses ``slogdet`` (not ``det``) so a large model matrix's determinant does not
    overflow/underflow. This is exactly the quantity the coordinate-exchange engine
    maximises, so it lives here and is shared. A singular (rank-deficient) design
    returns ``-inf``.
    """
    sign, logdet = np.linalg.slogdet(information_matrix(x))
    return float(logdet) if sign > 0 else float("-inf")


# --------------------------------------------------------------------------- #
# 1.2 Variance inflation & leverage
# --------------------------------------------------------------------------- #


def vif(x: np.ndarray, *, term_names: Sequence[str] | None = None) -> pd.Series:
    """Variance-inflation factor per non-intercept term.

    ``VIF_j = 1 / (1 - R_j^2)`` where term *j* is regressed on the others; computed in
    closed form from the correlation matrix's inverse diagonal rather than ``p`` separate
    regressions (falling back to per-term regressions only when exact collinearity makes
    that matrix singular, in which case fully aliased terms report ``inf``). ``VIF ~ 1``
    => orthogonal (no inflation); ``> 5-10`` => troublesome collinearity. The intercept --
    and any other constant column -- is excluded. ``term_names`` labels the returned
    series (defaults to positional names).
    """
    x = np.asarray(x, dtype=float)
    n_terms = x.shape[1]
    names = list(term_names) if term_names is not None else [f"x{j}" for j in range(n_terms)]
    if len(names) != n_terms:
        raise ValueError(f"term_names has {len(names)} entries for {n_terms} columns")

    # constant columns (the intercept, and any term that never varies) have no VIF
    keep = [j for j in range(n_terms) if np.std(x[:, j]) > 1e-12]
    cols = x[:, keep]
    kept_names = [names[j] for j in keep]

    if cols.shape[1] == 1:
        return pd.Series([1.0], index=kept_names, name="VIF")

    r = np.atleast_2d(np.corrcoef(cols, rowvar=False))
    try:
        vifs = np.diag(np.linalg.inv(r)).copy()
    except np.linalg.LinAlgError:
        # exact collinearity: R is singular, so invert term-by-term instead
        centered = cols - cols.mean(axis=0)
        vifs = np.empty(cols.shape[1])
        for j in range(cols.shape[1]):
            yj = centered[:, j]
            others = np.delete(centered, j, axis=1)
            resid = yj - others @ np.linalg.lstsq(others, yj, rcond=None)[0]
            ss_res = float(resid @ resid)
            ss_tot = float(yj @ yj)
            vifs[j] = float("inf") if ss_res <= 1e-12 * ss_tot else ss_tot / ss_res

    return pd.Series(vifs, index=kept_names, name="VIF")


def leverage(x: np.ndarray) -> np.ndarray:
    """Hat-matrix diagonal ``diag(H)``, ``H = X (X^T X)^-1 X^T`` -- leverage per run.

    Computed from the thin QR factor ``Q`` (``h_i = ||q_i||^2``), the same hat
    computation :func:`doe.analysis.anova.press` uses. ``sum(h_i) == p`` (number of
    model terms); a run with ``h == 1`` is fit exactly.
    """
    q, _ = np.linalg.qr(np.asarray(x, dtype=float))
    return np.asarray(np.einsum("ij,ij->i", q, q), dtype=float)


# --------------------------------------------------------------------------- #
# 1.3 Alias / correlation matrix (headless core re-homed from plotting)
# --------------------------------------------------------------------------- #


def correlation_matrix(x: np.ndarray, term_names: Sequence[str]) -> pd.DataFrame:
    """Pairwise Pearson correlations among model terms -- the design's alias structure.

    Constant columns (the intercept, and any squared pure-``+/-1`` term) have no defined
    correlation and are dropped. Off-diagonals near ``0`` mean terms are estimated
    independently (orthogonal); ``|r| = 1`` is full aliasing; intermediate magnitudes are
    partial aliasing (e.g. a Plackett-Burman main effect leaking ``+/-1/3`` into a two-factor
    interaction). This is the headless core that ``plotting.alias_matrix`` draws.
    """
    x = np.asarray(x, dtype=float)
    names = list(term_names)
    if len(names) != x.shape[1]:
        raise ValueError(f"term_names has {len(names)} entries for {x.shape[1]} columns")

    keep = [j for j in range(x.shape[1]) if np.std(x[:, j]) > 1e-12]
    kept_names = [names[j] for j in keep]
    corr = np.atleast_2d(np.corrcoef(x[:, keep], rowvar=False))
    return pd.DataFrame(corr, index=kept_names, columns=kept_names)


# --------------------------------------------------------------------------- #
# 1.4 Optimality efficiencies
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Efficiency:
    """D/A/G/I-efficiencies of a design, each normalised so an orthogonal design scores ~1.

    * ``d`` -- ``|X^T X|^(1/p) / n``; the headline number (full factorial => ``1.0``).
    * ``a`` -- normalised ``p / trace((X^T X)^-1)`` (average coefficient variance).
    * ``g`` -- from the *maximum* scaled prediction variance over the region.
    * ``i`` -- from the *average* (integrated) scaled prediction variance over the region.
    """

    d: float
    a: float
    g: float
    i: float


def _default_region(k: int, *, levels: int = 3, max_grid: int = 4096) -> np.ndarray:
    """A default candidate region over the coded box ``[-1, +1]^k``.

    The full ``levels``-level grid while it stays small (it contains the box corners, where
    prediction variance peaks, so the G-efficiency maximum is exact); a scrambled Sobol
    sample beyond that, where G/I become sample-based estimates.
    """
    if levels**k <= max_grid:
        axes = np.meshgrid(*([np.linspace(-1.0, 1.0, levels)] * k), indexing="ij")
        return np.column_stack([a.ravel() for a in axes])
    sampler = qmc.Sobol(d=k, scramble=True, seed=0)
    return np.asarray(2.0 * sampler.random(1024) - 1.0, dtype=float)


def _default_region_for_factors(
    factors: FactorSet, *, levels: int = 3, max_grid: int = 4096
) -> np.ndarray:
    """Default coded candidate region, preserving categorical factors' discrete levels."""
    if all(isinstance(factor, ContinuousFactor) for factor in factors):
        return _default_region(len(factors), levels=levels, max_grid=max_grid)

    axes: list[np.ndarray] = []
    for factor in factors:
        if isinstance(factor, ContinuousFactor):
            axes.append(np.linspace(-1.0, 1.0, levels))
        elif isinstance(factor, CategoricalFactor):
            axes.append(np.linspace(-1.0, 1.0, len(factor.levels)))

    n_grid = int(np.prod([len(axis) for axis in axes], dtype=np.int64))
    if n_grid <= max_grid:
        return np.asarray(list(itertools.product(*axes)), dtype=float)

    rng = np.random.default_rng(0)
    sampled = np.empty((1024, len(factors)), dtype=float)
    for j, axis in enumerate(axes):
        sampled[:, j] = rng.choice(axis, size=sampled.shape[0])
    return sampled


def efficiency(
    design: Design,
    *,
    order: int = 1,
    interactions: bool = True,
    region: np.ndarray | None = None,
) -> Efficiency:
    """D/A/G/I-efficiencies of ``design`` under the chosen model.

    ``order``/``interactions`` pick the model exactly as in
    :func:`~doe.analysis.model.build_model_matrix` (the diagnostics convention, matching
    ``anova_table``; the design-*building* generators in :mod:`doe.generators.optimal` take a
    ``model`` string instead, for ergonomics, like ``fit_ols``). The default is the
    first-order + interaction model; pass ``order=2`` to judge a response-surface design.

    G- and I-efficiency integrate the scaled prediction variance
    ``d(x) = n f(x)^T (X^T X)^-1 f(x)`` over a candidate ``region`` (an ``(m, k)`` array of
    coded points); when omitted, a grid (small ``k``) or Sobol sample of the coded box is
    used, so these are sample-based estimates, not closed-form integrals. I-efficiency is
    referenced against an ideal orthogonal design (``X^T X = n I``), for which ``d(x)`` is
    exactly ``f(x)^T f(x)``. The design and region rows are expanded through a *single*
    :func:`~doe.analysis.model.expand_coded_points` call so both share one term layout. For
    categorical factors, region columns use the discrete coordinates from
    ``np.linspace(-1, 1, n_levels)`` in natural level order. A design that cannot estimate the
    model (singular ``X^T X``) scores ``0`` across the board.
    """
    factors = design.factors
    coded = coded_design_points(design)
    n_runs = coded.shape[0]
    if region is None:
        region = _default_region_for_factors(factors)
    region = np.asarray(region, dtype=float)

    # one expansion for design + region rows => identical term layout for both
    expanded = expand_coded_points(
        np.vstack([coded, region]), factors, order=order, interactions=interactions
    )
    x, f_region = expanded.X[:n_runs], expanded.X[n_runs:]
    n_terms = x.shape[1]

    info = information_matrix(x)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        # a singular design leaves some coefficient direction with infinite variance
        return Efficiency(d=0.0, a=0.0, g=0.0, i=0.0)
    d_eff = float(np.exp(logdet / n_terms) / n_runs)

    info_inv = np.linalg.inv(info)
    a_eff = float(n_terms / (n_runs * np.trace(info_inv)))

    # scaled prediction variance at every region point
    spv = n_runs * np.einsum("ij,jk,ik->i", f_region, info_inv, f_region)
    g_eff = float(n_terms / spv.max())
    # orthogonal reference: with X^T X = n I the same quantity is just f(x)^T f(x)
    reference = np.einsum("ij,ij->i", f_region, f_region)
    i_eff = float(reference.mean() / spv.mean())

    return Efficiency(d=d_eff, a=a_eff, g=g_eff, i=i_eff)


# --------------------------------------------------------------------------- #
# Coverage metrics (Phase 4a)
# --------------------------------------------------------------------------- #
#
# Model-free companions to the model-based metrics above, for judging space-filling
# designs (and comparing any design's coverage). Both operate on ``Design.coded()``
# rescaled to the ``[0, 1]^k`` unit cube -- the convention ``scipy.stats.qmc`` expects.
# See ``docs/PHASE4.md`` section 1.2.


def _unit_cube_coords(design: Design) -> np.ndarray:
    """``Design.coded()`` (``[-1, +1]``) rescaled to the ``[0, 1]^k`` unit cube."""
    coded = np.asarray(design.coded().to_numpy(dtype=float))
    return (coded + 1.0) / 2.0


def discrepancy(
    design: Design, *, method: Literal["CD", "WD", "MD", "L2-star"] = "CD"
) -> float:
    """Uniformity of a design's coverage via ``scipy.stats.qmc.discrepancy``.

    Lower is more uniform. ``method`` is forwarded to scipy (``"CD"`` centered
    discrepancy by default).
    """
    return float(qmc.discrepancy(_unit_cube_coords(design), method=method))


def maximin_distance(design: Design) -> float:
    """Minimum pairwise distance between runs, in unit-cube coordinates.

    Larger means better-separated points; duplicated runs score ``0``.
    """
    return float(pdist(_unit_cube_coords(design)).min())
