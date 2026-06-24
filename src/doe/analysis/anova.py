"""ANOVA, lack-of-fit, and predictive R-squared for fitted models (Phase 2a).

Implemented in-house with sequential (Type I) sums of squares, computed from a QR
orthogonalisation of the model matrix (numerically stabler than repeated normal-equation
refits). F / t p-values come from ``scipy.stats``, which is already a core dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..design import Design
from .fit import FitResult


@dataclass
class LackOfFit:
    """Decomposition of residual variation into lack-of-fit and pure error.

    Pure error comes from replicated center points; a *non*-significant lack-of-fit
    (large ``p_value``) means the fitted model is adequate.
    """

    ss_lof: float
    df_lof: int
    ss_pe: float
    df_pe: int
    f_stat: float
    p_value: float


def _leverages(x: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix ``H = X (XtX)^-1 Xt`` via the (thin) QR factor Q."""
    q, _ = np.linalg.qr(x)
    return np.asarray(np.einsum("ij,ij->i", q, q), dtype=float)


def anova_table(result: FitResult, design: Design, response: np.ndarray) -> pd.DataFrame:
    """Sequential (Type I) ANOVA table for a fitted model.

    One row per non-intercept term (added in model order), plus ``Residual`` and ``Total``
    rows. Columns: ``SS, df, MS, F, p``. The term SS sum to the model SS, and the ``Total``
    SS equals the total corrected sum of squares.
    """
    del design  # signature symmetry with lack_of_fit; SS come from the model matrix
    y = np.asarray(response, dtype=float)
    x = result.model_matrix
    n_runs = x.shape[0]

    # Orthonormalise the columns in model order; for orthonormal Q the Type I SS of the
    # j-th column is just (q_j . y)^2 (the sequential reduction in residual SS).
    q, _ = np.linalg.qr(x)
    g = q.T @ y
    seq_ss = g**2  # one per column, intercept first

    ss_res = float(result.residuals @ result.residuals)
    dof_resid = result.dof_resid
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ms_resid = ss_res / dof_resid if dof_resid > 0 else float("nan")

    rows: dict[str, list[float]] = {"SS": [], "df": [], "MS": [], "F": [], "p": []}
    index: list[str] = []
    # skip the intercept (column 0): its SS is the mean-correction, not a model term
    for name, ss in zip(result.term_names[1:], seq_ss[1:], strict=True):
        # each term has 1 df, so its mean square equals its sum of squares. The F-ratio asks
        # whether the variation this term explains is large relative to the residual noise; the
        # p-value is the chance of an F that big if the term's true effect were zero.
        ms = float(ss)
        f = ms / ms_resid if ms_resid and np.isfinite(ms_resid) and ms_resid > 0 else float("nan")
        p = float(stats.f.sf(f, 1, dof_resid)) if np.isfinite(f) else float("nan")
        index.append(name)
        rows["SS"].append(float(ss))
        rows["df"].append(1.0)
        rows["MS"].append(ms)
        rows["F"].append(f)
        rows["p"].append(p)

    index.append("Residual")
    rows["SS"].append(ss_res)
    rows["df"].append(float(dof_resid))
    rows["MS"].append(ms_resid)
    rows["F"].append(float("nan"))
    rows["p"].append(float("nan"))

    index.append("Total")
    rows["SS"].append(ss_tot)
    rows["df"].append(float(n_runs - 1))
    rows["MS"].append(float("nan"))
    rows["F"].append(float("nan"))
    rows["p"].append(float("nan"))

    return pd.DataFrame(rows, index=index)


def lack_of_fit(result: FitResult, design: Design, response: np.ndarray) -> LackOfFit:
    """Lack-of-fit test against pure error from replicated center points.

    Requires at least two center points (so pure error has >= 1 degree of freedom).
    """
    y = np.asarray(response, dtype=float)
    center = design.center_indices
    if len(center) < 2:
        raise ValueError("lack-of-fit needs at least 2 replicated center points for pure error")

    # Pure error: replicate runs share identical factor settings, so any spread among their
    # responses is pure experimental noise -- a model-free yardstick for the residual variance.
    y_center = y[center]
    ss_pe = float(((y_center - y_center.mean()) ** 2).sum())
    df_pe = len(center) - 1

    # Lack of fit is whatever residual variation is left after removing pure error: variation
    # the model failed to capture (e.g. missing curvature). The F-test below compares the two;
    # a significant result means the model is inadequate, not merely that the data are noisy.
    ss_res = float(result.residuals @ result.residuals)
    df_resid = result.dof_resid
    ss_lof = ss_res - ss_pe
    df_lof = df_resid - df_pe
    if df_lof <= 0:
        raise ValueError("no degrees of freedom left for lack-of-fit")

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pe / df_pe
    f_stat = ms_lof / ms_pe if ms_pe > 0 else float("inf")
    p_value = float(stats.f.sf(f_stat, df_lof, df_pe)) if np.isfinite(f_stat) else 0.0
    return LackOfFit(ss_lof, df_lof, ss_pe, df_pe, f_stat, p_value)


def press(result: FitResult) -> float:
    """PRESS statistic ``sum((e_i / (1 - h_i))**2)`` from leave-one-out residuals.

    PRESS measures how well the model *predicts runs it did not see*: dividing each residual by
    ``1 - h_i`` rescales it into the residual the model would have had if that run were left out
    of the fit (an algebraic shortcut that avoids ``n`` refits). It penalises over-fitting that
    ordinary R^2 rewards, and underlies the predicted R^2 below.
    """
    h = _leverages(result.model_matrix)
    denom = 1.0 - h
    return float(np.sum((result.residuals / denom) ** 2))


def predicted_r2(result: FitResult) -> float:
    """Predicted R-squared (Q-squared) ``1 - PRESS / SS_tot``."""
    y = result.fitted + result.residuals
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - press(result) / ss_tot


def adjusted_r2(result: FitResult) -> float:
    """Adjusted R-squared ``1 - (1 - R2)(n - 1)/(n - p)``.

    Plain R^2 can only rise as terms are added; adjusting by the degrees of freedom charges for
    each extra parameter, so it falls when a term explains less than chance would. A large gap
    between R^2 and adjusted R^2 is a sign the model is padded with unhelpful terms.
    """
    n_runs = result.model_matrix.shape[0]
    n_terms = len(result.term_names)
    if n_runs - n_terms <= 0:
        return float("nan")
    return float(1.0 - (1.0 - result.r_squared) * (n_runs - 1) / (n_runs - n_terms))
