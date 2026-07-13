"""Restricted maximum likelihood (REML) for the two-stratum split-plot model (Phase 5b).

The split-plot model is a mixed model ``y = Xβ + Zγ + ε`` with whole-plot random effects
``γ ~ N(0, σ²_wp I)`` and sub-plot error ``ε ~ N(0, σ² I)``; ``Z`` is the whole-plot indicator.
The response covariance is block-diagonal per whole plot,

    ``V = σ² I + σ²_wp Z Zᵀ = σ² (I + η Z Zᵀ)``,   ``η = σ²_wp / σ²``,

so it is governed by the single non-negative ratio ``η``. REML profiles the log-likelihood over
``η`` (a 1-D bounded search); ``σ̂²`` is closed-form at each step and ``V₀⁻¹ = (I + η Z Zᵀ)⁻¹`` is
block-diagonal, each block inverted exactly via Sherman-Morrison (``(I + η J)⁻¹ = I − η/(1 + η
n_p) J``). Headless and dependency-free beyond numpy/scipy -- the ``diagnostics.py`` pattern.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar


def _plot_groups(whole_plots: Sequence[int]) -> list[np.ndarray]:
    """Row-index arrays, one per whole plot, in first-appearance order."""
    plots = list(whole_plots)
    return [
        np.array([i for i, p in enumerate(plots) if p == g], dtype=int)
        for g in dict.fromkeys(plots)
    ]


def v0_inverse(eta: float, whole_plots: Sequence[int]) -> np.ndarray:
    """``V₀⁻¹ = (I + η Z Zᵀ)⁻¹`` for the whole-plot indicator ``Z``, built block by block.

    Each whole plot contributes an ``n_p × n_p`` block ``I − η/(1 + η n_p) J`` (Sherman-Morrison);
    blocks for different plots do not interact, so the full inverse is block-diagonal.
    """
    n = len(whole_plots)
    v_inv = np.zeros((n, n))
    for idx in _plot_groups(whole_plots):
        n_p = len(idx)
        block = np.eye(n_p) - (eta / (1.0 + eta * n_p)) * np.ones((n_p, n_p))
        v_inv[np.ix_(idx, idx)] = block
    return v_inv


def reml_variance_components(
    x: np.ndarray, y: np.ndarray, whole_plots: Sequence[int]
) -> tuple[float, float, float]:
    """Estimate ``(σ²_wp, σ², REML log-likelihood)`` for a split-plot fit by REML.

    ``x`` is the model matrix, ``y`` the response, ``whole_plots`` the per-run plot ids. The
    profiled REML objective in ``η`` is minimized over ``log η`` with a bounded scalar search;
    ``σ̂²`` and ``β̂`` are the closed-form GLS quantities at the optimum.
    """
    groups = _plot_groups(whole_plots)
    plot_sizes = [len(g) for g in groups]
    n, p = x.shape
    dof = n - p
    if dof <= 0:
        raise ValueError("split-plot REML needs residual degrees of freedom (n > n_terms)")

    def profiled_neg2ll(eta: float) -> tuple[float, float]:
        """Return ``(-2 ℓ_R up to a constant, σ̂²)`` at this ``η``."""
        v_inv = v0_inverse(eta, whole_plots)
        xtvi = x.T @ v_inv
        a = xtvi @ x
        rhs = xtvi @ y
        beta = np.linalg.solve(a, rhs)
        y_p_y = float(y @ (v_inv @ y) - rhs @ beta)
        sigma2 = y_p_y / dof
        logdet_v0 = float(sum(np.log1p(eta * m) for m in plot_sizes))
        _sign, logdet_a = np.linalg.slogdet(a)
        return dof * np.log(sigma2) + logdet_v0 + float(logdet_a), sigma2

    result = minimize_scalar(
        lambda t: profiled_neg2ll(float(np.exp(t)))[0],
        bounds=(np.log(1e-8), np.log(1e8)),
        method="bounded",
    )
    eta_hat = float(np.exp(result.x))
    neg2ll, sigma2 = profiled_neg2ll(eta_hat)
    sigma2_wp = eta_hat * sigma2
    reml_loglik = -0.5 * (neg2ll + dof * np.log(2.0 * np.pi) + dof)
    return sigma2_wp, sigma2, reml_loglik
