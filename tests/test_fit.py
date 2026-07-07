import numpy as np
import pandas as pd
import pytest

from doe.analysis import anova
from doe.analysis.fit import FitResult, fit_ols
from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet, MixtureFactor
from doe.generators.factorial import full_factorial
from doe.generators.mixture import simplex_lattice


def test_fit_recovers_known_effects():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()

    # response built in coded units: y = 10 + 3*a + 2*b + 1.5*a*b
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b

    result = fit_ols(design, y, order=1, interactions=True)
    summary = result.summary()

    assert np.isclose(summary["Intercept"][0], 10.0)
    # effect = 2 * coefficient in coded units
    assert np.isclose(summary["a"][1], 6.0)
    assert np.isclose(summary["b"][1], 4.0)
    assert np.isclose(summary["a:b"][1], 3.0)
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
        assert np.isclose(summary[name][0], coef)
    # effect = 2 * coefficient holds for the 2-level categorical contrast too
    assert np.isclose(summary["catalyst[B]"][1], 4.0)
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
    assert pred.shape == (1,)
    assert pred[0] == pytest.approx(result.coefficients[0])


def test_predict_raises_when_a_required_term_cannot_be_produced():
    # a quadratic fit needs an "a^2" column; scoring only cube-corner points can't emit it
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=3)
    y = np.arange(design.n_runs, dtype=float)
    result = fit_ols(design, y, model="quadratic")
    with pytest.raises(ValueError, match="required model term"):
        result.predict({"a": [0.0, 10.0], "b": [0.0, 10.0]})


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
