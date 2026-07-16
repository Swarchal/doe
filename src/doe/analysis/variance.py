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


def v0_inv_products(
    eta: float, x: np.ndarray, y: np.ndarray, whole_plots: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The GLS pieces ``(XᵀV₀⁻¹X, XᵀV₀⁻¹y, yᵀV₀⁻¹y, log|V₀|)``, accumulated per whole plot.

    ``V₀⁻¹`` is block-diagonal, each whole plot's ``n_p × n_p`` block being
    ``I − c J`` with ``c = η/(1 + η n_p)`` (Sherman-Morrison, ``J`` the all-ones block);
    blocks for different plots do not interact. Contracting a block against that plot's rows
    ``Xₚ``/``yₚ`` collapses onto their column sums ``sₚ = Xₚᵀ1`` and ``tₚ = 1ᵀyₚ``:

        ``Xₚᵀ(I − cJ)Xₚ = XₚᵀXₚ − c sₚsₚᵀ``,   ``Xₚᵀ(I − cJ)yₚ = Xₚᵀyₚ − c tₚ sₚ``,
        ``yₚᵀ(I − cJ)yₚ = yₚᵀyₚ − c tₚ²``,      ``log|V₀| = Σₚ log(1 + η n_p)``.

    So every quantity the GLS/REML fit needs is summed plot-by-plot in ``O(n p²)`` time and
    ``O(p²)`` memory, never forming the dense ``n × n`` inverse (which the direct
    ``x.T @ V₀⁻¹`` product would allocate at ``O(n²)`` and multiply at ``O(p n²)``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = x.shape[1]
    xtvix = np.zeros((p, p))
    xtviy = np.zeros(p)
    ytviy = 0.0
    logdet_v0 = 0.0
    for idx in _plot_groups(whole_plots):
        n_p = len(idx)
        xp = x[idx]
        yp = y[idx]
        s = xp.sum(axis=0)  # column sums = Xₚᵀ 1
        t = float(yp.sum())  # 1ᵀ yₚ
        c = eta / (1.0 + eta * n_p)
        xtvix += xp.T @ xp - c * np.outer(s, s)
        xtviy += xp.T @ yp - c * t * s
        ytviy += float(yp @ yp) - c * t * t
        logdet_v0 += float(np.log1p(eta * n_p))
    return xtvix, xtviy, ytviy, logdet_v0


def reml_variance_components(
    x: np.ndarray, y: np.ndarray, whole_plots: Sequence[int]
) -> tuple[float, float, float]:
    """Estimate ``(σ²_wp, σ², REML log-likelihood)`` for a split-plot fit by REML.

    ``x`` is the model matrix, ``y`` the response, ``whole_plots`` the per-run plot ids. The
    profiled REML objective in ``η`` is minimized over ``log η`` with a bounded scalar search;
    ``σ̂²`` and ``β̂`` are the closed-form GLS quantities at the optimum.
    """
    n, p = x.shape
    dof = n - p
    if dof <= 0:
        raise ValueError("split-plot REML needs residual degrees of freedom (n > n_terms)")

    def profiled_neg2ll(eta: float) -> tuple[float, float]:
        """Return ``(-2 ℓ_R up to a constant, σ̂²)`` at this ``η``."""
        a, rhs, ytviy, logdet_v0 = v0_inv_products(eta, x, y, whole_plots)
        beta = np.linalg.solve(a, rhs)
        y_p_y = ytviy - float(rhs @ beta)
        sigma2 = y_p_y / dof
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
