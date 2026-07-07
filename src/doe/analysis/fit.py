"""Ordinary-least-squares fitting and factor-effect estimates."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..design import Design
from ..factors import FactorSet
from .model import build_model_matrix

if TYPE_CHECKING:
    from .anova import LackOfFit
    from .optimize import Bounds, Optimum, StationaryPoint

ModelSpec = Literal["linear", "quadratic", "scheffe-linear", "scheffe-quadratic"]

#: Convenience model names -> ``(order, interactions)``.
_MODEL_SPECS: dict[str, tuple[int, bool]] = {
    "linear": (1, True),
    "quadratic": (2, True),
    "scheffe-linear": (1, False),
    "scheffe-quadratic": (2, False),
}


@dataclass
class FitResult:
    """The outcome of fitting a linear model to a design + response.

    Examples:
        >>> from doe import ContinuousFactor, FactorSet, fit_ols, full_factorial
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     ContinuousFactor("time", 5, 15),
        ... ])
        >>> design = full_factorial(factors)
        >>> coded = design.coded()
        >>> response = 10 + 2 * coded["temperature"] - coded["time"]
        >>> fit = fit_ols(design, response, interactions=False)
        >>> tuple(round(v, 6) for v in fit.summary()["temperature"])
        (2.0, 4.0)
        >>> fit.predict({"temperature": [40, 80], "time": 10}).round(6).tolist()
        [8.0, 12.0]
    """

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
    order: int
    interactions: bool
    #: The design this result was fitted from (``None`` when constructed directly, not via
    #: :func:`fit_ols`). The fluent post-fit methods need it.
    design: Design | None = None
    #: The resolved response array the fit used, aligned to the design's runs (``None`` when
    #: constructed directly). Stashed so :meth:`anova`/:meth:`lack_of_fit` need no re-passing.
    response: np.ndarray | None = None

    def _require_source(self) -> tuple[Design, np.ndarray]:
        """Return the stashed ``(design, response)`` or explain why they are missing."""
        if self.design is None or self.response is None:
            raise ValueError(
                "this method needs a FitResult produced by fit_ols (which stashes the "
                "originating design and response); this result was constructed directly"
            )
        return self.design, self.response

    def anova(self) -> pd.DataFrame:
        """Sequential (Type I) ANOVA table for this fit (see :func:`analysis.anova`)."""
        from . import anova

        design, response = self._require_source()
        return anova.anova_table(self, design, response)

    def lack_of_fit(self) -> LackOfFit:
        """Lack-of-fit test against pure error (see :func:`analysis.anova.lack_of_fit`)."""
        from . import anova

        design, response = self._require_source()
        return anova.lack_of_fit(self, design, response)

    def press(self) -> float:
        """PRESS statistic from leave-one-out residuals (see :func:`analysis.anova.press`)."""
        from . import anova

        return anova.press(self)

    def predicted_r2(self) -> float:
        """Predicted R-squared / Q-squared (see :func:`analysis.anova.predicted_r2`)."""
        from . import anova

        return anova.predicted_r2(self)

    def adjusted_r2(self) -> float:
        """Adjusted R-squared (see :func:`analysis.anova.adjusted_r2`)."""
        from . import anova

        return anova.adjusted_r2(self)

    def predict(self, points: Mapping[str, object] | pd.DataFrame | Design) -> np.ndarray:
        """Predict responses at new runs given in *natural* units.

        ``points`` is a mapping (column name -> scalar or array), a :class:`pandas.DataFrame`,
        or a :class:`~doe.design.Design`. Its factor columns are coded through the stored
        :class:`~doe.factors.FactorSet` and expanded with the *same* term structure the fit used
        (``order``/``interactions`` and, for a mixture fit, the Scheffé blending path), then dotted
        with the fitted coefficients.

        The expanded columns are aligned to the fit's ``term_names`` *by name* before the dot
        product: :func:`~doe.analysis.model.expand_coded_points` only emits a squared term once a
        column takes a value off ``+/-1``, so new points sitting entirely on the cube corners would
        otherwise silently drop terms and misalign. A required term the new points cannot produce
        raises rather than returning a wrong number.
        """
        from ..design import Design as _Design
        from .model import coded_design_points, expand_coded_points

        frame = _points_to_frame(points, self.factors.names)
        design = _Design(frame, self.factors)
        coded_points = coded_design_points(design)
        mm = expand_coded_points(
            coded_points, self.factors, order=self.order, interactions=self.interactions
        )
        available = dict(zip(mm.term_names, mm.X.T, strict=True))
        missing = [term for term in self.term_names if term not in available]
        if missing:
            raise ValueError(
                f"the supplied points do not produce required model term(s) {missing}; "
                "predict cannot align them to the fitted coefficients"
            )
        x_aligned = np.column_stack([available[term] for term in self.term_names])
        return np.asarray(x_aligned @ self.coefficients, dtype=float)

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

    def stationary_point(self) -> StationaryPoint:
        """Unconstrained stationary point of the fitted surface (see :func:`optimize`)."""
        from .optimize import stationary_point

        return stationary_point(self)

    def optimum(
        self, *, maximize: bool = True, bounds: Bounds = (-1.0, 1.0)
    ) -> Optimum:
        """Constrained optimum over the coded design box (see :func:`optimize.optimum`)."""
        from .optimize import optimum

        return optimum(self, maximize=maximize, bounds=bounds)


def _points_to_frame(
    points: Mapping[str, object] | pd.DataFrame | Design, names: list[str]
) -> pd.DataFrame:
    """Coerce ``predict`` input to a natural-unit frame with the factor columns present."""
    if isinstance(points, Design):
        frame: pd.DataFrame | None = points.runs
        mapping: dict[str, Any] = {}
    elif isinstance(points, pd.DataFrame):
        frame = points
        mapping = {}
    else:
        frame = None
        mapping = dict(points)

    source_keys = frame.columns if frame is not None else mapping.keys()
    missing = [name for name in names if name not in source_keys]
    if missing:
        raise ValueError(f"points missing factor column(s) {missing}")

    if frame is not None:
        return frame.loc[:, names].reset_index(drop=True)

    # mapping path: broadcast scalars against any array-valued columns of a shared length
    arrays: dict[str, np.ndarray] = {name: np.asarray(mapping[name]) for name in names}
    lengths = {arr.shape[0] for arr in arrays.values() if arr.ndim > 0}
    if len(lengths) > 1:
        raise ValueError(f"array-valued columns must share a length; got lengths {sorted(lengths)}")
    n = lengths.pop() if lengths else 1
    return pd.DataFrame({name: np.broadcast_to(arr, n) for name, arr in arrays.items()})


def fit_ols(
    design: Design,
    response: np.ndarray | str,
    *,
    order: int = 1,
    interactions: bool = True,
    model: ModelSpec | None = None,
) -> FitResult:
    """Fit an OLS model in coded units and return coefficients and factor effects.

    ``response`` is either the measured values (array-like, aligned to the runs) or the *name*
    of a response column already on the design -- the pairing produced by
    :meth:`Design.with_response`, which is the safer path since it cannot silently misalign.

    In coded (+/-1) units the *effect* of a term is twice its regression coefficient --
    the change in response moving a factor from -1 to +1.

    ``model`` is a convenience over ``order``/``interactions``: ``"linear"`` == ``order=1,
    interactions=True`` and ``"quadratic"`` == ``order=2, interactions=True``.

    **Mixture designs** (all-:class:`~doe.factors.MixtureFactor` factor sets) are fitted with
    Scheffé blending models -- no intercept, ``order=1`` for linear blending, ``order=2`` for
    quadratic (the ``"scheffe-linear"``/``"scheffe-quadratic"`` model names make the intent
    explicit and additionally *require* a mixture design). Two conventions to know:

    * ``r_squared`` stays *centered* (against ``sum((y - mean(y))^2)``). This is valid for
      Scheffé models -- the proportions sum to 1, so the constant lies in the model's column
      space and residuals still sum to zero -- and it keeps R^2 comparable with intercept
      models, avoiding the inflated "uncorrected" R^2 that no-intercept fits report in some
      packages (the statsmodels gotcha).
    * ``effects`` is all-NaN: the classic factorial effect is the -1 -> +1 coded swing
      (``2 x coefficient``), which has no meaning for blending coefficients on proportions.

    Examples:
        Fit a factorial design from a response column attached to the design.

        >>> from doe import ContinuousFactor, FactorSet, fit_ols, full_factorial
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     ContinuousFactor("time", 5, 15),
        ... ])
        >>> design = full_factorial(factors)
        >>> coded = design.coded()
        >>> response = 50 + 3 * coded["temperature"] + 2 * coded["time"]
        >>> fit = fit_ols(design.with_response("yield", response), "yield", interactions=False)
        >>> fit.term_names
        ['Intercept', 'temperature', 'time']
        >>> fit.coefficients.round(6).tolist()
        [50.0, 3.0, 2.0]
        >>> fit.effects.round(6).tolist()
        [50.0, 6.0, 4.0]

        Use the named Scheffé models for all-mixture designs.

        >>> import pandas as pd
        >>> from doe import Design, MixtureFactor
        >>> blend_factors = FactorSet([MixtureFactor("A"), MixtureFactor("B")])
        >>> blend = Design(pd.DataFrame({"A": [1.0, 0.0, 0.5], "B": [0.0, 1.0, 0.5]}),
        ...                blend_factors)
        >>> blend_fit = fit_ols(blend, [12.0, 18.0, 15.0], model="scheffe-linear")
        >>> blend_fit.term_names
        ['A', 'B']
        >>> blend_fit.coefficients.round(6).tolist()
        [12.0, 18.0]
    """
    if model is not None:
        if model not in _MODEL_SPECS:
            raise ValueError(f"unknown model {model!r}; expected one of {sorted(_MODEL_SPECS)}")
        if model.startswith("scheffe") and not design.factors.is_mixture:
            raise ValueError(
                f"model {model!r} requires an all-mixture design "
                "(every factor a MixtureFactor)"
            )
        order, interactions = _MODEL_SPECS[model]

    if isinstance(response, str):
        if response not in design.runs.columns:
            raise ValueError(
                f"no response column {response!r} on the design; "
                f"available columns: {list(design.runs.columns)}"
            )
        response = design.runs[response].to_numpy()

    y = np.asarray(response, dtype=float)
    if y.shape[0] != design.n_runs:
        raise ValueError("response length must match number of runs")

    mm = build_model_matrix(design, order=order, interactions=interactions)
    x = mm.X
    # least-squares solution of X b = y; in a balanced/orthogonal design the coefficients are
    # exactly the half-effects, but lstsq also handles the non-orthogonal (e.g. CCD) case.
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)

    fitted = x @ coef
    residuals = y - fitted
    # R^2 is the fraction of the total (mean-corrected) variation in the response explained by
    # the model: 1 - unexplained/total. Undefined when the response never varies.
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # residual degrees of freedom = runs minus parameters estimated; this is the budget that
    # pays for the error variance and therefore for every standard error and p-value below.
    n_runs, n_terms = x.shape
    dof_resid = n_runs - n_terms
    xtx_inv = np.linalg.pinv(x.T @ x)

    if dof_resid > 0:
        # mean squared error estimates the experimental noise variance; scaling (X'X)^-1 by it
        # gives the coefficient covariance, whose diagonal square-roots are the standard errors.
        mse = ss_res / dof_resid
        cov_beta = mse * xtx_inv
        std_errors = np.sqrt(np.diag(cov_beta))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = coef / std_errors
            p_values = 2.0 * stats.t.sf(np.abs(t_values), dof_resid)
    else:
        # a saturated model spends every run on a parameter, leaving nothing to estimate noise
        # with; effects can still be computed but their significance cannot be judged. (Such a
        # design is usually read with a half-normal plot instead -- see plotting.half_normal_plot.)
        warnings.warn(
            "model is saturated (residual dof = 0); standard errors are undefined",
            stacklevel=2,
        )
        mse = float("nan")
        cov_beta = np.full((n_terms, n_terms), np.nan)
        std_errors = np.full(n_terms, np.nan)
        t_values = np.full(n_terms, np.nan)
        p_values = np.full(n_terms, np.nan)

    # an "effect" is the response change over the full -1 -> +1 swing of a coded factor, i.e.
    # twice the coefficient (the slope per coded unit). This is the classic factorial-effect
    # scale that the Pareto/half-normal plots and most DoE textbooks report. Scheffé blending
    # coefficients live on proportions with no -1 -> +1 swing, so mixture fits get NaN.
    if design.factors.is_mixture:
        effects = np.full_like(coef, np.nan)
    else:
        effects = 2.0 * coef
        effects[0] = coef[0]  # intercept is the grand mean, not a swing -- leave it untouched
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
        order,
        interactions,
        design,
        y,
    )
