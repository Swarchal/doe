"""Ordinary-least-squares fitting and factor-effect estimates."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from ..design import Design
from ..factors import FactorSet
from .model import build_model_matrix

ModelSpec = Literal["linear", "quadratic"]

#: Convenience model names -> ``(order, interactions)``.
_MODEL_SPECS: dict[str, tuple[int, bool]] = {
    "linear": (1, True),
    "quadratic": (2, True),
}


@dataclass
class FitResult:
    """The outcome of fitting a linear model to a design + response."""

    term_names: list[str]
    coefficients: np.ndarray
    effects: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    r_squared: float
    model_matrix: np.ndarray
    dof_resid: int
    mse: float
    cov_beta: np.ndarray
    std_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    factors: FactorSet

    def summary(self) -> dict[str, tuple[float, float]]:
        """Map each term to ``(coefficient, effect)``."""
        return {
            name: (float(c), float(e))
            for name, c, e in zip(self.term_names, self.coefficients, self.effects, strict=True)
        }

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        """Two-sided confidence interval per coefficient as an ``(n_terms, 2)`` array."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")
        if self.dof_resid <= 0:
            half = np.full_like(self.coefficients, np.nan)
        else:
            t_crit = float(stats.t.ppf(0.5 + level / 2.0, self.dof_resid))
            half = t_crit * self.std_errors
        return np.column_stack([self.coefficients - half, self.coefficients + half])


def fit_ols(
    design: Design,
    response: np.ndarray,
    *,
    order: int = 1,
    interactions: bool = True,
    model: ModelSpec | None = None,
) -> FitResult:
    """Fit an OLS model in coded units and return coefficients and factor effects.

    In coded (+/-1) units the *effect* of a term is twice its regression coefficient --
    the change in response moving a factor from -1 to +1.

    ``model`` is a convenience over ``order``/``interactions``: ``"linear"`` == ``order=1,
    interactions=True`` and ``"quadratic"`` == ``order=2, interactions=True``.
    """
    if model is not None:
        if model not in _MODEL_SPECS:
            raise ValueError(f"unknown model {model!r}; expected one of {sorted(_MODEL_SPECS)}")
        order, interactions = _MODEL_SPECS[model]

    y = np.asarray(response, dtype=float)
    if y.shape[0] != design.n_runs:
        raise ValueError("response length must match number of runs")

    mm = build_model_matrix(design, order=order, interactions=interactions)
    x = mm.X
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)

    fitted = x @ coef
    residuals = y - fitted
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_runs, n_terms = x.shape
    dof_resid = n_runs - n_terms
    xtx_inv = np.linalg.pinv(x.T @ x)

    if dof_resid > 0:
        mse = ss_res / dof_resid
        cov_beta = mse * xtx_inv
        std_errors = np.sqrt(np.diag(cov_beta))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = coef / std_errors
            p_values = 2.0 * stats.t.sf(np.abs(t_values), dof_resid)
    else:
        warnings.warn(
            "model is saturated (residual dof = 0); standard errors are undefined",
            stacklevel=2,
        )
        mse = float("nan")
        cov_beta = np.full((n_terms, n_terms), np.nan)
        std_errors = np.full(n_terms, np.nan)
        t_values = np.full(n_terms, np.nan)
        p_values = np.full(n_terms, np.nan)

    effects = 2.0 * coef
    effects[0] = coef[0]  # intercept has no "effect" interpretation
    return FitResult(
        mm.term_names,
        coef,
        effects,
        fitted,
        residuals,
        r_squared,
        x,
        dof_resid,
        mse,
        cov_beta,
        std_errors,
        t_values,
        p_values,
        design.factors,
    )
