"""Ordinary-least-squares fitting and factor-effect estimates."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import pandas as pd
from scipy import stats

from ..design import Design
from ..factors import FactorSet
from ..serialization import json_safe
from .model import build_model_matrix

if TYPE_CHECKING:
    from .anova import LackOfFit
    from .optimize import Bounds, Optimum, StationaryPoint

ModelSpec = Literal["linear", "quadratic", "scheffe-linear", "scheffe-quadratic"]


class RankDeficientModelError(ValueError):
    """Raised by :func:`fit_ols`/:func:`fit_gls` when the design cannot estimate the model.

    The model matrix has fewer independent columns than terms, so at least one term is an
    exact linear combination of the others -- a resolution-III fraction fitted with
    two-factor interactions, say, where a main effect *is* an interaction column. Least
    squares still returns *a* solution (the minimum-norm one, which splits an aliased
    effect evenly between the confounded columns), so the failure is silent unless caught
    here: the reported effects would be a fraction of their true size and phantom terms
    would appear significant.
    """


class SaturatedFitWarning(UserWarning):
    """Raised by :func:`fit_ols` when a model spends every run on a parameter.

    A saturated fit (``dof_resid == 0``) has no degrees of freedom left to estimate
    noise, so standard errors/t/p-values are undefined (NaN). A distinct category --
    rather than a bare ``UserWarning`` -- lets callers (the web service, in particular)
    map the condition to a stable string (``"saturated_model"``) by ``category``, not by
    matching the message text.
    """

#: Convenience model names -> ``(order, interactions)``. Public because it is also the
#: web service's wire vocabulary for ``model`` (``docs/WEBSERVICE_API.md``); the service
#: resolves against this table rather than restating it.
MODEL_SPECS: dict[str, tuple[int, bool]] = {
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
        >>> fit.summary().loc["temperature", ["coefficient", "effect"]].round(6).tolist()
        [2.0, 4.0]
        >>> fit.predict({"temperature": [40, 80], "time": 10}).round(6).tolist()
        [8.0, 12.0]
        >>> round(fit.predict({"temperature": 40, "time": 10}), 6)
        8.0
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
    #: The response column *name*, when ``fit_ols`` was called with ``response`` as a string
    #: (``None`` for an array-valued response, or a directly constructed result). Reserved for
    #: labelling multi-response output (e.g. a future ``desirability`` report); not yet read
    #: anywhere.
    response_name: str | None = None
    #: Whole-plot variance component ``σ²_wp`` from a :func:`fit_gls` split-plot fit (``None`` for
    #: an OLS fit). ``mse`` holds the sub-plot variance ``σ²`` for such a fit.
    sigma2_wp: float | None = None
    #: Number of whole plots in a :func:`fit_gls` fit (``None`` for OLS).
    n_whole_plots: int | None = None
    #: Per-term residual degrees of freedom for a two-stratum :func:`fit_gls` fit (``None`` for
    #: OLS, where the single scalar ``dof_resid`` applies to every term). Whole-plot terms test
    #: against whole-plot df, sub-plot terms against ``dof_resid``; :meth:`conf_int` uses it.
    dof_terms: np.ndarray | None = None

    def _require_source(self) -> tuple[Design, np.ndarray]:
        """Return the stashed ``(design, response)`` or explain why they are missing."""
        if self.design is None or self.response is None:
            raise ValueError(
                "this method needs a FitResult produced by fit_ols (which stashes the "
                "originating design and response); this result was constructed directly"
            )
        return self.design, self.response

    def _require_single_stratum(self, what: str) -> None:
        """Refuse an OLS-only statistic on a two-stratum (split-plot) fit.

        ``dof_terms`` is set only by :func:`fit_gls`, whose whole-plot and sub-plot errors are
        separate strata. Every statistic below is built on a *single* pooled residual variance,
        so on a split-plot fit it would silently test whole-plot terms against the (much smaller)
        sub-plot error -- reintroducing the exact anticonservative bias ``fit_gls`` exists to
        remove. Refusing beats answering wrongly.
        """
        if self.dof_terms is not None:
            raise NotImplementedError(
                f"{what} is not defined for a split-plot (GLS) fit: it assumes one pooled error "
                "stratum, but this fit has two (whole-plot and sub-plot). Use summary() or "
                "conf_int(), which honour the per-term degrees of freedom in dof_terms."
            )

    def anova(self) -> pd.DataFrame:
        """Sequential (Type I) ANOVA table for this fit (see :func:`analysis.anova`)."""
        from . import anova

        self._require_single_stratum("anova")
        design, response = self._require_source()
        return anova.anova_table(self, design, response)

    def lack_of_fit(self) -> LackOfFit:
        """Lack-of-fit test against pure error (see :func:`analysis.anova.lack_of_fit`)."""
        from . import anova

        self._require_single_stratum("lack_of_fit")
        design, response = self._require_source()
        return anova.lack_of_fit(self, design, response)

    def press(self) -> float:
        """PRESS statistic from leave-one-out residuals (see :func:`analysis.anova.press`)."""
        from . import anova

        self._require_single_stratum("press")
        return anova.press(self)

    def predicted_r2(self) -> float:
        """Predicted R-squared / Q-squared (see :func:`analysis.anova.predicted_r2`)."""
        from . import anova

        self._require_single_stratum("predicted_r2")
        return anova.predicted_r2(self)

    def adjusted_r2(self) -> float:
        """Adjusted R-squared (see :func:`analysis.anova.adjusted_r2`)."""
        from . import anova

        return anova.adjusted_r2(self)

    @overload
    def predict(
        self,
        points: Mapping[str, object] | pd.DataFrame | Design,
        *,
        interval: None = ...,
        level: float = ...,
    ) -> np.ndarray | float: ...

    @overload
    def predict(
        self,
        points: Mapping[str, object] | pd.DataFrame | Design,
        *,
        interval: Literal["confidence", "prediction"],
        level: float = ...,
    ) -> pd.DataFrame: ...

    def predict(
        self,
        points: Mapping[str, object] | pd.DataFrame | Design,
        *,
        interval: Literal["confidence", "prediction"] | None = None,
        level: float = 0.95,
    ) -> np.ndarray | float | pd.DataFrame:
        """Predict responses at new runs given in *natural* units.

        ``points`` is a mapping (column name -> scalar or array), a :class:`pandas.DataFrame`,
        or a :class:`~doe.design.Design`. Its factor columns are coded through the stored
        :class:`~doe.factors.FactorSet` and expanded with the *same* term structure the fit used
        (``order``/``interactions`` and, for a mixture fit, the Scheffé blending path), then dotted
        with the fitted coefficients.

        The expanded columns are aligned to the fit's ``term_names`` *by name* before the dot
        product. :func:`~doe.analysis.model.expand_coded_points` normally only emits a squared
        column once a factor's values sit off ``+/-1`` (the right heuristic at fit time), which
        would otherwise drop squared terms for new points sitting entirely on the cube corners
        (e.g. a single natural low/high run). Here the required squared terms are derived
        straight from the fit's own ``term_names`` and forced via ``require_squares``, so a
        quadratic fit predicts correctly even at a single corner point. A required term the new
        points still cannot produce (e.g. an unknown categorical level) raises rather than
        returning a wrong number.

        With ``interval=None`` (default) this returns the point predictions only: a plain
        ``float`` when ``points`` is a mapping whose values are *all* scalars (a single
        natural-unit run) -- the common single-point case, so callers don't have to unwrap a
        length-1 array -- and an ``ndarray`` (one entry per run) for any other input.

        Passing ``interval`` instead returns a :class:`pandas.DataFrame` (one row per point,
        regardless of input shape) with columns ``fit``/``se``/``lower``/``upper`` at the given
        confidence ``level``. ``"confidence"`` bounds the *mean* response at each setting
        (variance ``xᵀ cov(β) x``); ``"prediction"`` bounds a *single future observation* there,
        widening the band by the residual variance (``mse + xᵀ cov(β) x``) -- the interval a
        confirmation run should land inside. Both use a two-sided Student-``t`` multiplier on
        the fit's residual degrees of freedom; ``se``/``lower``/``upper`` are NaN for a
        saturated model (``dof_resid == 0``, no error variance to build a band from), matching
        :meth:`conf_int`.
        """
        from ..design import Design as _Design
        from .model import coded_design_points, expand_coded_points

        scalar_input = isinstance(points, Mapping) and all(
            np.asarray(value).ndim == 0 for value in points.values()
        )

        frame = _points_to_frame(points, self.factors.names)
        design = _Design(frame, self.factors)
        coded_points = coded_design_points(design)
        require_squares = [t[:-2] for t in self.term_names if t.endswith("^2")]
        mm = expand_coded_points(
            coded_points,
            self.factors,
            order=self.order,
            interactions=self.interactions,
            require_squares=require_squares,
        )
        available = dict(zip(mm.term_names, mm.X.T, strict=True))
        missing = [term for term in self.term_names if term not in available]
        if missing:
            raise ValueError(
                f"the supplied points do not produce required model term(s) {missing}; "
                "predict cannot align them to the fitted coefficients"
            )
        x_aligned = np.column_stack([available[term] for term in self.term_names])
        predicted = np.asarray(x_aligned @ self.coefficients, dtype=float)
        if interval is None:
            return float(predicted[0]) if scalar_input else predicted
        # ``str(...)`` so this guard stays reachable for untyped callers passing a bad string.
        if str(interval) not in ("confidence", "prediction"):
            raise ValueError("interval must be 'confidence', 'prediction', or None")
        return self._prediction_interval(x_aligned, predicted, interval, level)

    def _prediction_interval(
        self,
        x_aligned: np.ndarray,
        predicted: np.ndarray,
        interval: Literal["confidence", "prediction"],
        level: float,
    ) -> pd.DataFrame:
        """Build the ``fit``/``se``/``lower``/``upper`` band for :meth:`predict`."""
        self._require_single_stratum("predict(interval=...)")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")
        # Var of the mean response at each point: diag(X cov(beta) Xᵀ).
        mean_var = np.einsum("ij,jk,ik->i", x_aligned, self.cov_beta, x_aligned)
        if interval == "prediction":
            mean_var = mean_var + self.mse
        if self.dof_resid <= 0:
            se = np.full_like(predicted, np.nan)
            half = np.full_like(predicted, np.nan)
        else:
            se = np.sqrt(mean_var)
            t_crit = float(stats.t.ppf(0.5 + level / 2.0, self.dof_resid))
            half = t_crit * se
        return pd.DataFrame(
            {"fit": predicted, "se": se, "lower": predicted - half, "upper": predicted + half}
        )

    def summary(self) -> pd.DataFrame:
        """Coefficient/effect/inference table, one row per term.

        Columns: ``coefficient``, ``effect``, ``std_error``, ``t``, ``p``. The last three
        come straight from the fit's inference arrays and are NaN for a saturated model
        (see :func:`fit_ols`).

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
            >>> fit.summary().loc["temperature", ["coefficient", "effect"]].round(6).tolist()
            [2.0, 4.0]
        """
        return pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "effect": self.effects,
                "std_error": self.std_errors,
                "t": self.t_values,
                "p": self.p_values,
            },
            index=pd.Index(self.term_names, name="term"),
        )

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Two-sided confidence interval per coefficient, one row per term.

        Columns ``lower``/``upper``. NaN throughout when the model is saturated
        (``dof_resid == 0``, so there is no error variance to build an interval from).
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")
        if self.dof_terms is not None:
            # two-stratum split-plot fit: each term uses its own (whole-plot or sub-plot) df
            dof = np.asarray(self.dof_terms, dtype=float)
            with np.errstate(invalid="ignore"):
                t_crit = np.where(
                    dof > 0, stats.t.ppf(0.5 + level / 2.0, np.where(dof > 0, dof, 1.0)), np.nan
                )
            half = t_crit * self.std_errors
        elif self.dof_resid <= 0:
            half = np.full_like(self.coefficients, np.nan)
        else:
            t_crit_scalar = float(stats.t.ppf(0.5 + level / 2.0, self.dof_resid))
            half = t_crit_scalar * self.std_errors
        return pd.DataFrame(
            {"lower": self.coefficients - half, "upper": self.coefficients + half},
            index=pd.Index(self.term_names, name="term"),
        )

    def vif(self) -> pd.Series:
        """Variance-inflation factor per term (see :func:`diagnostics.vif`)."""
        from . import diagnostics

        return diagnostics.vif(self.model_matrix, term_names=self.term_names)

    def leverage(self) -> np.ndarray:
        """Hat-matrix diagonal per run (see :func:`diagnostics.leverage`)."""
        from . import diagnostics

        return diagnostics.leverage(self.model_matrix)

    def stationary_point(self) -> StationaryPoint:
        """Unconstrained stationary point of the fitted surface (see :func:`optimize`)."""
        from .optimize import stationary_point

        return stationary_point(self)

    def optimum(self, *, maximize: bool = True, bounds: Bounds = (-1.0, 1.0)) -> Optimum:
        """Constrained optimum over the coded design box (see :func:`optimize.optimum`)."""
        from .optimize import optimum

        return optimum(self, maximize=maximize, bounds=bounds)

    def to_dict(self, *, confidence: float = 0.95) -> dict[str, Any]:
        """Serialize to the ``POST /v1/analysis/fit`` response body (minus ``warnings``).

        ``terms`` is one record per model term (``term``, ``coefficient``, ``effect``,
        ``std_error``, ``t``, ``p``, ``ci_low``, ``ci_high``), built from :meth:`summary`
        and :meth:`conf_int` -- no statistic is recomputed here. A saturated fit's NaN
        inference columns, and a Scheffé (mixture) fit's all-NaN ``effect`` column, come
        through :func:`~doe.serialization.json_safe` as ``null``, matching the
        ``docs/WEBSERVICE_API.md`` contract. ``model`` echoes the resolved
        ``(order, interactions)`` spec this fit used.

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
            >>> payload = fit.to_dict()
            >>> payload["terms"][1]["term"], round(payload["terms"][1]["effect"], 6)
            ('temperature', 4.0)
        """
        summary = self.summary()
        ci = self.conf_int(confidence)
        terms = [
            {
                "term": term,
                "coefficient": summary.loc[term, "coefficient"],
                "effect": summary.loc[term, "effect"],
                "std_error": summary.loc[term, "std_error"],
                "t": summary.loc[term, "t"],
                "p": summary.loc[term, "p"],
                "ci_low": ci.loc[term, "lower"],
                "ci_high": ci.loc[term, "upper"],
            }
            for term in self.term_names
        ]
        return cast(
            "dict[str, Any]",
            json_safe(
                {
                    "terms": terms,
                    "r_squared": self.r_squared,
                    "adjusted_r2": self.adjusted_r2(),
                    "dof_resid": self.dof_resid,
                    "mse": self.mse,
                    "fitted": self.fitted,
                    "residuals": self.residuals,
                    "model": {"order": self.order, "interactions": self.interactions},
                }
            ),
        )


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


def _check_estimable(x: np.ndarray, term_names: list[str]) -> None:
    """Raise if the design cannot estimate every model term independently.

    Each column is admitted in model order and the rank is re-checked: a column that does
    not raise the rank is an exact linear combination of the ones before it, i.e. aliased
    with them. Reporting *those* columns (rather than a bare "singular matrix") points at
    the terms to drop -- for a resolution-III fraction they are precisely the interactions
    the design was never able to resolve.
    """
    rank = np.linalg.matrix_rank(x)
    if rank == x.shape[1]:
        return

    aliased: list[str] = []
    kept = 0  # rank of the columns admitted so far
    for j in range(x.shape[1]):
        if np.linalg.matrix_rank(x[:, : j + 1]) > kept:
            kept += 1
        else:
            aliased.append(term_names[j])

    n_runs, n_terms = x.shape
    raise RankDeficientModelError(
        f"the design cannot estimate this model: the model matrix has {n_terms} terms but "
        f"rank {rank} ({n_runs} runs), so term(s) {aliased} are exactly aliased with earlier "
        "terms and their effects cannot be separated. Fit a smaller model (e.g. "
        "interactions=False for a resolution-III fraction, or order=1 to drop squared terms), "
        "or run a design that supports the terms you want."
    )


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
        if model not in MODEL_SPECS:
            raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODEL_SPECS)}")
        if model.startswith("scheffe") and not design.factors.is_mixture:
            raise ValueError(
                f"model {model!r} requires an all-mixture design (every factor a MixtureFactor)"
            )
        order, interactions = MODEL_SPECS[model]

    response_name: str | None = None
    if isinstance(response, str):
        if response not in design.runs.columns:
            raise ValueError(
                f"no response column {response!r} on the design; "
                f"available columns: {list(design.runs.columns)}"
            )
        response_name = response
        response = design.runs[response].to_numpy()

    y = np.asarray(response, dtype=float)
    if y.shape[0] != design.n_runs:
        raise ValueError("response length must match number of runs")

    mm = build_model_matrix(design, order=order, interactions=interactions)
    x = mm.X
    # lstsq happily returns the minimum-norm solution for an aliased model, splitting a
    # confounded effect between its aliases instead of failing -- so rule that out first.
    _check_estimable(x, mm.term_names)
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
            SaturatedFitWarning,
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
        response_name,
    )


def _term_factor_names(term: str) -> set[str]:
    """The base factor names a model term references (empty for the intercept).

    ``"temperature:catalyst[B]"`` -> ``{"temperature", "catalyst"}``; ``"time^2"`` -> ``{"time"}``.
    """
    if term == "Intercept":
        return set()
    names: set[str] = set()
    for part in term.split(":"):
        base = part.split("[", maxsplit=1)[0]
        if base.endswith("^2"):
            base = base[:-2]
        names.add(base)
    return names


def fit_gls(
    design: Design,
    response: np.ndarray | str,
    *,
    order: int = 1,
    interactions: bool = True,
    model: ModelSpec | None = None,
) -> FitResult:
    """Fit a split-plot design by generalized least squares with REML variance components.

    A split-plot design (``design.whole_plots is not None``) has two error strata -- whole-plot
    and sub-plot -- so OLS is wrong: it pools them into one error and *understates* whole-plot
    standard errors (the classic anticonservative split-plot trap). ``fit_gls`` estimates the
    variance ratio by REML (:func:`~doe.analysis.variance.reml_variance_components`), then returns
    the GLS estimates ``β̂ = (XᵀV⁻¹X)⁻¹XᵀV⁻¹y`` with ``Cov(β̂) = (XᵀV⁻¹X)⁻¹``.

    The model matrix is built exactly as in :func:`fit_ols` (same ``order``/``interactions``/
    ``model``), and the result is the **same** :class:`FitResult` type, so every downstream
    consumer reads it unchanged. It additionally carries the variance components
    (:attr:`~FitResult.sigma2_wp`, ``mse`` = sub-plot ``σ²``, :attr:`~FitResult.n_whole_plots`)
    and per-term degrees of freedom (:attr:`~FitResult.dof_terms`) following the containment rule:
    a term built only from hard-to-change factors (and the intercept) is tested against whole-plot
    df (``n_plots − #whole-plot-level terms``); every other term against the sub-plot residual df
    (the scalar :attr:`~FitResult.dof_resid`). Effects keep their ``2 × coefficient`` meaning
    (factors are coded to ``[-1, +1]``, unlike the mixture/Scheffé path).

    Raises:
        ValueError: when ``design.whole_plots is None`` (use :func:`fit_ols`), for a mixture
            design (split-plot × mixture is out of scope), or a Scheffé ``model``.
    """
    if design.whole_plots is None:
        raise ValueError(
            "fit_gls requires a split-plot design (design.whole_plots is not None); "
            "a fully-randomized design is fitted with fit_ols"
        )
    if design.factors.is_mixture:
        raise ValueError("fit_gls does not support mixture designs")
    if model is not None:
        if model not in MODEL_SPECS:
            raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODEL_SPECS)}")
        if model.startswith("scheffe"):
            raise ValueError("fit_gls does not support Scheffé (mixture) models")
        order, interactions = MODEL_SPECS[model]

    from .variance import reml_variance_components, v0_inverse

    response_name: str | None = None
    if isinstance(response, str):
        if response not in design.runs.columns:
            raise ValueError(
                f"no response column {response!r} on the design; "
                f"available columns: {list(design.runs.columns)}"
            )
        response_name = response
        response = design.runs[response].to_numpy()
    y = np.asarray(response, dtype=float)
    if y.shape[0] != design.n_runs:
        raise ValueError("response length must match number of runs")

    mm = build_model_matrix(design, order=order, interactions=interactions)
    x = mm.X
    _check_estimable(x, mm.term_names)
    whole_plots = design.whole_plots

    sigma2_wp, sigma2, _reml_ll = reml_variance_components(x, y, whole_plots)
    eta = sigma2_wp / sigma2 if sigma2 > 0 else 0.0
    v_inv = v0_inverse(eta, whole_plots)

    xtvi = x.T @ v_inv
    xtvix_inv = np.linalg.pinv(xtvi @ x)
    coef = xtvix_inv @ (xtvi @ y)
    cov_beta = sigma2 * xtvix_inv
    std_errors = np.sqrt(np.diag(cov_beta))

    fitted = x @ coef
    residuals = y - fitted
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # two-stratum degrees of freedom by the containment rule
    wp_names = {f.name for f in design.factors.whole_plot_factors}
    is_wp_term = np.array(
        [_term_factor_names(term) <= wp_names for term in mm.term_names], dtype=bool
    )
    n_plots = len(set(whole_plots))
    n_runs = x.shape[0]
    n_wp_terms = int(is_wp_term.sum())
    n_sp_terms = len(mm.term_names) - n_wp_terms
    dof_wp = n_plots - n_wp_terms
    dof_sp = n_runs - n_plots - n_sp_terms
    dof_terms = np.where(is_wp_term, dof_wp, dof_sp).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = coef / std_errors
        p_values = np.array(
            [
                2.0 * stats.t.sf(abs(t), d) if d > 0 else np.nan
                for t, d in zip(t_values, dof_terms, strict=True)
            ]
        )

    effects = 2.0 * coef
    effects[0] = coef[0]  # intercept is the grand mean, not a swing
    return FitResult(
        mm.term_names,
        coef,
        effects,
        fitted,
        residuals,
        r_squared,
        x,
        int(dof_sp),
        sigma2,
        cov_beta,
        std_errors,
        t_values,
        p_values,
        design.factors,
        order,
        interactions,
        design,
        y,
        response_name,
        sigma2_wp=sigma2_wp,
        n_whole_plots=n_plots,
        dof_terms=dof_terms,
    )
