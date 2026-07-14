import numpy as np
import pandas as pd
import pytest
from scipy import stats

from doe.analysis import anova, diagnostics
from doe.analysis.fit import (
    FitResult,
    RankDeficientModelError,
    SaturatedFitWarning,
    fit_ols,
)
from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet, MixtureFactor
from doe.generators.factorial import fractional_factorial, full_factorial
from doe.generators.mixture import simplex_lattice
from doe.generators.rsm import central_composite


def test_fit_recovers_known_effects():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()

    # response built in coded units: y = 10 + 3*a + 2*b + 1.5*a*b
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b

    result = fit_ols(design, y, order=1, interactions=True)
    summary = result.summary()

    assert np.isclose(summary.loc["Intercept", "coefficient"], 10.0)
    # effect = 2 * coefficient in coded units
    assert np.isclose(summary.loc["a", "effect"], 6.0)
    assert np.isclose(summary.loc["b", "effect"], 4.0)
    assert np.isclose(summary.loc["a:b", "effect"], 3.0)
    assert np.isclose(result.r_squared, 1.0)


def test_fit_recovers_effects_with_categorical_factor():
    temp = ContinuousFactor("temp", 0, 100)
    catalyst = CategoricalFactor("catalyst", ("A", "B"))
    design = full_factorial([temp, catalyst])  # 4 corner runs

    # build a noiseless response from known coefficients on the coded model matrix
    mm = build_model_matrix(design, order=1, interactions=True)
    # term order: Intercept, temp, catalyst[B], temp:catalyst[B]
    beta = {"Intercept": 10.0, "temp": 3.0, "catalyst[B]": 2.0, "temp:catalyst[B]": 1.5}
    y = mm.X @ np.array([beta[name] for name in mm.term_names])

    result = fit_ols(design, y, order=1, interactions=True)
    summary = result.summary()

    for name, coef in beta.items():
        assert np.isclose(summary.loc[name, "coefficient"], coef)
    # effect = 2 * coefficient holds for the 2-level categorical contrast too
    assert np.isclose(summary.loc["catalyst[B]", "effect"], 4.0)
    assert np.isclose(result.r_squared, 1.0)


def test_fit_records_model_spec():
    # the resolved (order, interactions) must be stored so a serialized FitResult can be
    # re-fitted -- the result object alone otherwise can't say how it was built.
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    y = np.arange(design.n_runs, dtype=float)

    result = fit_ols(design, y, order=2, interactions=False)
    assert result.order == 2
    assert result.interactions is False

    # the recorded spec round-trips: re-fitting with it reproduces the term names exactly
    refit = fit_ols(design, y, order=result.order, interactions=result.interactions)
    assert refit.term_names == result.term_names


def test_fit_records_model_spec_from_named_model():
    # the convenience ``model="quadratic"`` resolves to (order=2, interactions=True);
    # the resolved values, not the name, are what gets recorded.
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=3)
    y = np.arange(design.n_runs, dtype=float)

    result = fit_ols(design, y, model="quadratic")
    assert result.order == 2
    assert result.interactions is True


def test_fit_response_length_mismatch():
    design = full_factorial([ContinuousFactor("a", 0, 1)], levels=2)
    try:
        fit_ols(design, np.array([1.0, 2.0, 3.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched response length")


def _replicated_design_and_response():
    """A 2^2 factorial replicated twice (residual dof > 0, avoids the saturated warning)."""
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2).replicate(2)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b + np.array([0.3, -0.2, 0.1, -0.1, -0.3, 0.2, -0.1, 0.1])
    return design, y


# --------------------------------------------------------------------------- #
# Phase B: fluent post-fit analysis
# --------------------------------------------------------------------------- #


def test_fit_ols_stashes_design_and_response():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    assert result.design is design
    assert np.allclose(result.response, y)


def test_result_anova_matches_free_function():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    pd.testing.assert_frame_equal(result.anova(), anova.anova_table(result, design, y))


def test_result_lack_of_fit_matches_free_function():
    design, y = _replicated_design_and_response()
    # drop the interaction so a lack-of-fit degree of freedom is left over
    result = fit_ols(design, y, interactions=False)
    fluent = result.lack_of_fit()
    direct = anova.lack_of_fit(result, design, y)
    assert fluent == direct


def test_result_predictive_metrics_match_free_functions():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    assert result.press() == pytest.approx(anova.press(result))
    assert result.predicted_r2() == pytest.approx(anova.predicted_r2(result))
    assert result.adjusted_r2() == pytest.approx(anova.adjusted_r2(result))


def test_predict_reproduces_fitted_on_training_points():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    # accepts a Design, a DataFrame, and a dict of columns -- all reproduce ``fitted``
    assert np.allclose(result.predict(design), result.fitted)
    assert np.allclose(result.predict(design.runs), result.fitted)
    natural = {name: design.runs[name].to_numpy() for name in design.factors.names}
    assert np.allclose(result.predict(natural), result.fitted)


def test_predict_quadratic_reproduces_fitted():
    # a 3-level design exercises the off-+/-1 squared-term path in expand_coded_points
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=3)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 5 + 2 * a - b + 1.5 * a * b - 0.7 * a**2 + 0.4 * b**2
    result = fit_ols(design, y, model="quadratic")
    assert np.allclose(result.predict(design.runs), result.fitted)


def test_predict_single_point_as_scalar_dict():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    pred = result.predict({"a": 5.0, "b": 5.0})  # coded center -> intercept
    assert isinstance(pred, float)
    assert pred == pytest.approx(result.coefficients[0])


def test_predict_scalar_dict_matches_array_path():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    scalar = result.predict({"a": 10.0, "b": 5.0})
    batched = result.predict({"a": [10.0], "b": [5.0]})
    assert isinstance(scalar, float)
    assert isinstance(batched, np.ndarray)
    assert scalar == pytest.approx(batched[0])


def test_predict_dataframe_and_design_stay_arrays():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    assert isinstance(result.predict(design), np.ndarray)
    assert isinstance(result.predict(design.runs), np.ndarray)


def test_predict_single_corner_point_quadratic():
    # a lone cube-corner point (coded +/-1 on every factor) can't trigger the off-+/-1
    # heuristic expand_coded_points normally uses to decide whether to emit a squared
    # column; predict must still produce the right answer for a quadratic fit.
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = central_composite(factors)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 5 + 2 * a - b + 1.5 * a * b - 0.7 * a**2 + 0.4 * b**2
    result = fit_ols(design, y, model="quadratic")

    corner = result.predict({"a": 10.0, "b": 10.0})
    assert isinstance(corner, float)

    # manual evaluation at coded (1, 1) using the fit's own recovered coefficients
    beta = result.summary()["coefficient"]
    expected = (
        beta["Intercept"]
        + beta["a"] * 1.0
        + beta["b"] * 1.0
        + beta["a:b"] * 1.0
        + beta["a^2"] * 1.0
        + beta["b^2"] * 1.0
    )
    assert corner == pytest.approx(expected)

    # must agree with the old workaround of batching the corner with an interior point
    batched = result.predict({"a": [10.0, 5.0], "b": [10.0, 5.0]})
    assert batched[0] == pytest.approx(corner)


def test_predict_honors_scheffe_mixture_path():
    components = [
        MixtureFactor("x1"),
        MixtureFactor("x2"),
        MixtureFactor("x3"),
    ]
    design = simplex_lattice(components, degree=2)
    coded = design.coded().to_numpy()
    x1, x2, x3 = coded[:, 0], coded[:, 1], coded[:, 2]
    y = 3 * x1 + 5 * x2 + 2 * x3 + 4 * x1 * x2 - x1 * x3 + 2 * x2 * x3
    result = fit_ols(design, y, model="scheffe-quadratic")
    assert np.allclose(result.predict(design.runs), result.fitted)


def test_predict_interval_returns_frame_with_fit_bracketed():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    band = result.predict({"a": 10.0, "b": 5.0}, interval="prediction")
    assert isinstance(band, pd.DataFrame)
    assert list(band.columns) == ["fit", "se", "lower", "upper"]
    assert len(band) == 1
    # point estimate matches the plain (interval-free) prediction, and the band brackets it
    point = result.predict({"a": 10.0, "b": 5.0})
    assert band.loc[0, "fit"] == pytest.approx(point)
    assert band.loc[0, "lower"] < point < band.loc[0, "upper"]


def test_prediction_interval_wider_than_confidence_by_residual_variance():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    point = {"a": 8.0, "b": 3.0}
    conf = result.predict(point, interval="confidence")
    pred = result.predict(point, interval="prediction")
    # same centre; prediction band is strictly wider (adds the residual variance term)
    assert pred.loc[0, "fit"] == pytest.approx(conf.loc[0, "fit"])
    assert pred.loc[0, "se"] > conf.loc[0, "se"]
    assert pred.loc[0, "lower"] < conf.loc[0, "lower"]
    assert pred.loc[0, "upper"] > conf.loc[0, "upper"]
    # prediction variance == confidence (mean) variance + residual MSE
    assert pred.loc[0, "se"] ** 2 == pytest.approx(conf.loc[0, "se"] ** 2 + result.mse)


def test_prediction_interval_matches_closed_form():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    level = 0.95
    band = result.predict({"a": 10.0, "b": 10.0}, interval="prediction", level=level)
    # x row for the corner (1, 1): intercept, a, b, a:b in coded units
    x = np.array([1.0, 1.0, 1.0, 1.0])
    mean_var = float(x @ result.cov_beta @ x)
    se = np.sqrt(mean_var + result.mse)
    t_crit = stats.t.ppf(0.5 + level / 2.0, result.dof_resid)
    fit_val = float(x @ result.coefficients)
    assert band.loc[0, "se"] == pytest.approx(se)
    assert band.loc[0, "lower"] == pytest.approx(fit_val - t_crit * se)
    assert band.loc[0, "upper"] == pytest.approx(fit_val + t_crit * se)


def test_predict_interval_batches_one_row_per_point():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    band = result.predict({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]}, interval="confidence")
    assert isinstance(band, pd.DataFrame)
    assert len(band) == 3
    assert np.all(band["lower"] < band["upper"])


def test_prediction_interval_nan_when_saturated():
    # a 2^2 factorial with the full interaction model has zero residual dof
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b
    with pytest.warns(UserWarning):
        result = fit_ols(design, y, model="linear")
    assert result.dof_resid == 0
    band = result.predict({"a": 10.0, "b": 5.0}, interval="prediction")
    # fit is still a real number; the band is undefined (no error variance)
    assert np.isfinite(band.loc[0, "fit"])
    assert np.isnan(band.loc[0, "se"])
    assert np.isnan(band.loc[0, "lower"])
    assert np.isnan(band.loc[0, "upper"])


def test_predict_interval_rejects_bad_arguments():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    with pytest.raises(ValueError, match="interval must be"):
        result.predict({"a": 5.0, "b": 5.0}, interval="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level must be between 0 and 1"):
        result.predict({"a": 5.0, "b": 5.0}, interval="prediction", level=1.5)


def test_fluent_methods_require_a_fit_ols_result():
    # directly constructed FitResult has no stashed design/response
    result = FitResult(
        term_names=["Intercept"],
        coefficients=np.array([1.0]),
        effects=np.array([1.0]),
        fitted=np.array([1.0]),
        residuals=np.array([0.0]),
        r_squared=1.0,
        model_matrix=np.ones((1, 1)),
        dof_resid=0,
        mse=float("nan"),
        cov_beta=np.array([[np.nan]]),
        std_errors=np.array([np.nan]),
        t_values=np.array([np.nan]),
        p_values=np.array([np.nan]),
        factors=FactorSet([ContinuousFactor("a", 0, 1)]),
        order=1,
        interactions=True,
    )
    with pytest.raises(ValueError, match="produced by fit_ols"):
        result.anova()
    with pytest.raises(ValueError, match="produced by fit_ols"):
        result.lack_of_fit()


# --------------------------------------------------------------------------- #
# summary() / conf_int() as labeled frames
# --------------------------------------------------------------------------- #


def test_summary_returns_frame_with_expected_shape_and_columns():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    summary = result.summary()

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == result.term_names
    assert list(summary.columns) == ["coefficient", "effect", "std_error", "t", "p"]
    assert np.allclose(summary["coefficient"].to_numpy(), result.coefficients)
    assert np.allclose(summary["effect"].to_numpy(), result.effects)
    assert np.allclose(summary["std_error"].to_numpy(), result.std_errors)
    assert np.allclose(summary["t"].to_numpy(), result.t_values)
    assert np.allclose(summary["p"].to_numpy(), result.p_values)


def test_conf_int_returns_frame_matching_manual_computation():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = central_composite(factors, center=5)
    rng = np.random.default_rng(1)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 50 + 3 * a - 2 * b + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    ci = result.conf_int(level=0.95)
    assert isinstance(ci, pd.DataFrame)
    assert list(ci.index) == result.term_names
    assert list(ci.columns) == ["lower", "upper"]
    assert np.all(ci["lower"].to_numpy() <= result.coefficients)
    assert np.all(result.coefficients <= ci["upper"].to_numpy())

    from scipy import stats as _stats

    t_crit = float(_stats.t.ppf(0.975, result.dof_resid))
    half = t_crit * result.std_errors
    assert np.allclose(ci["lower"].to_numpy(), result.coefficients - half)
    assert np.allclose(ci["upper"].to_numpy(), result.coefficients + half)


# --------------------------------------------------------------------------- #
# fluent vif()/leverage()
# --------------------------------------------------------------------------- #


def test_fit_vif_matches_free_function():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    pd.testing.assert_series_equal(
        result.vif(), diagnostics.vif(result.model_matrix, term_names=result.term_names)
    )


def test_fit_leverage_matches_free_function():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    assert np.allclose(result.leverage(), diagnostics.leverage(result.model_matrix))


# --------------------------------------------------------------------------- #
# response_name
# --------------------------------------------------------------------------- #


def test_response_name_stashed_for_string_response():
    design, y = _replicated_design_and_response()
    named = design.with_response("yield", y)
    result = fit_ols(named, "yield")
    assert result.response_name == "yield"


def test_response_name_none_for_array_response():
    design, y = _replicated_design_and_response()
    result = fit_ols(design, y)
    assert result.response_name is None


# --------------------------------------------------------------------------- #
# rank deficiency (an aliased model must not be "fitted" silently)
# --------------------------------------------------------------------------- #


def test_fit_raises_on_aliased_model_rather_than_halving_effects():
    # A resolution-III fraction aliases C with A:B. Asking for interactions (the *default*)
    # gives 7 terms for 4 runs: lstsq would return the minimum-norm solution, splitting A's
    # true effect evenly between A and its alias B:C -- every effect halved, a phantom
    # interaction, and dof_resid = -3. It must raise instead.
    factors = FactorSet([ContinuousFactor(n, 0, 1) for n in "ABC"])
    design = fractional_factorial(factors, generators=["C=AB"])
    y = 1 + 2 * design.coded()["A"].to_numpy()

    with pytest.raises(RankDeficientModelError, match="exactly aliased"):
        fit_ols(design, y)  # order=1, interactions=True


def test_aliased_model_error_names_the_inestimable_terms():
    factors = FactorSet([ContinuousFactor(n, 0, 1) for n in "ABC"])
    design = fractional_factorial(factors, generators=["C=AB"])
    y = 1 + 2 * design.coded()["A"].to_numpy()

    with pytest.raises(RankDeficientModelError) as exc:
        fit_ols(design, y)
    message = str(exc.value)
    # the two-factor interactions are what this design cannot resolve
    for term in ("A:B", "A:C", "B:C"):
        assert term in message


def test_correct_model_on_the_same_fraction_recovers_the_true_effect():
    # the fix must not cost the *estimable* model anything: main effects only, effect = 2 x coef
    factors = FactorSet([ContinuousFactor(n, 0, 1) for n in "ABC"])
    design = fractional_factorial(factors, generators=["C=AB"])
    y = 1 + 2 * design.coded()["A"].to_numpy()

    result = fit_ols(design, y, interactions=False)
    assert result.summary().loc["A", "effect"] == pytest.approx(4.0)
    assert result.dof_resid == 0


def test_saturated_but_full_rank_model_still_fits():
    # dof_resid == 0 is saturated, not rank-deficient: it must still fit (with the warning)
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()
    y = 10 + 3 * coded[:, 0] + 2 * coded[:, 1] + 1.5 * coded[:, 0] * coded[:, 1]

    with pytest.warns(SaturatedFitWarning):
        result = fit_ols(design, y, order=1, interactions=True)
    assert result.dof_resid == 0
    assert result.summary().loc["a", "effect"] == pytest.approx(6.0)
