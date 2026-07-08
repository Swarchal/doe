"""Response-surface optimization.

Written test-first to document the intended API and the numbers behind each result.
The anchors are quadratics built in coded units on a face-centered CCD (which supports
the full second-order model), so the fit recovers the coefficients exactly and the
stationary-point / optimum maths is checkable against closed-form values.
"""

import numpy as np
import pandas as pd
import pytest

from doe.analysis.fit import fit_ols
from doe.analysis.optimize import (
    ResponseGoal,
    desirability,
    optimum,
    stationary_point,
)
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import full_factorial
from doe.generators.rsm import central_composite


def _ccd(center=5):
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    return central_composite(factors, center=center)


def _fit(coded_response, *, model="quadratic", order=2, interactions=True, response_name=None):
    """Fit a response defined as a function of the coded factor columns (x1, x2)."""
    design = _ccd()
    coded = design.coded().to_numpy()
    y = coded_response(coded[:, 0], coded[:, 1])
    if response_name is not None:
        design = design.with_response(response_name, y)
        response: np.ndarray | str = response_name
    else:
        response = y
    if model is None:
        return fit_ols(design, response, order=order, interactions=interactions)
    return fit_ols(design, response, model=model)


# --------------------------------------------------------------------------- #
# Stationary point + canonical analysis
# --------------------------------------------------------------------------- #


def test_stationary_point_recovers_interior_maximum():
    # concave quadratic; stationary point solves grad = b + 2Bx = 0 -> x = -1/2 B^-1 b.
    # b = [3, 2], B = [[-4, -1/2], [-1/2, -3]]  =>  x_s = [16/47, 13/47].
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    sp = stationary_point(result)

    assert np.allclose(sp.coded, [16.0 / 47.0, 13.0 / 47.0])
    assert sp.kind == "maximum"
    assert np.all(sp.eigenvalues < 0)

    # decoded to natural units: a, b each span [0, 10] (center 5, half-range 5)
    assert np.isclose(sp.natural["a"], 5.0 + 5.0 * 16.0 / 47.0)
    assert np.isclose(sp.natural["b"], 5.0 + 5.0 * 13.0 / 47.0)


def _mixed_fit():
    """A fit including a categorical factor -- not optimizable as a continuous surface."""
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        ContinuousFactor("time", 0.0, 10.0),
        CategoricalFactor("cat", ("A", "B")),
    ]
    design = full_factorial(factors, levels=2)
    rng = np.random.default_rng(0)
    return fit_ols(design, rng.normal(size=design.n_runs), order=1, interactions=True)


def test_stationary_point_rejects_categorical_factor():
    # surface optimization is undefined over a discrete factor; expect a clear TypeError
    # (regression guard: previously raised an opaque KeyError 'cat[B]')
    result = _mixed_fit()
    with pytest.raises(TypeError, match="continuous"):
        stationary_point(result)


def test_optimum_rejects_categorical_factor():
    result = _mixed_fit()
    with pytest.raises(TypeError, match="continuous"):
        optimum(result)


def test_desirability_rejects_categorical_factor():
    result = _mixed_fit()
    goal = ResponseGoal(result, goal="max", low=-5.0, high=5.0)
    with pytest.raises(TypeError, match="continuous"):
        desirability([goal])


def test_stationary_point_predicted_response_is_the_extremum():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    sp = stationary_point(result)

    # the predicted response at the stationary point must dominate the whole coded box
    grid = np.linspace(-1.0, 1.0, 41)
    gx, gy = np.meshgrid(grid, grid)
    surface = (
        50.0 + 3.0 * gx + 2.0 * gy - 4.0 * gx**2 - 3.0 * gy**2 - 1.0 * gx * gy
    )
    assert sp.response >= surface.max() - 1e-6


def test_stationary_point_classifies_minimum():
    # convex quadratic (B positive definite) -> interior minimum
    result = _fit(
        lambda x1, x2: 50.0 - 2.0 * x1 + 4.0 * x2 + 3.0 * x1**2 + 2.0 * x2**2 + 1.0 * x1 * x2
    )
    sp = stationary_point(result)
    assert sp.kind == "minimum"
    assert np.all(sp.eigenvalues > 0)


def test_stationary_point_classifies_saddle():
    # opposite-sign curvatures -> saddle
    result = _fit(lambda x1, x2: 50.0 + 4.0 * x1**2 - 3.0 * x2**2)
    sp = stationary_point(result)
    assert sp.kind == "saddle"
    assert np.any(sp.eigenvalues > 0) and np.any(sp.eigenvalues < 0)


def test_stationary_point_requires_curvature():
    result = _fit(lambda x1, x2: 50.0 + 2.0 * x1 + 3.0 * x2, model=None, order=1)
    with pytest.raises(ValueError):
        stationary_point(result)


# --------------------------------------------------------------------------- #
# Constrained optimum over the coded box
# --------------------------------------------------------------------------- #


def test_optimum_interior_matches_stationary_point():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    opt = optimum(result, maximize=True)
    sp = stationary_point(result)

    assert np.allclose(opt.coded, sp.coded, atol=1e-4)
    assert opt.at_bound is False
    assert np.isclose(opt.response, sp.response, atol=1e-5)


def test_optimum_clamps_to_box_when_stationary_point_outside():
    # stationary point at x1 = 5 (outside [-1, 1]); surface increases across the box in
    # both factors, so the constrained maximum sits at the (1, 1) corner.
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    opt = optimum(result, maximize=True)

    assert np.allclose(opt.coded, [1.0, 1.0], atol=1e-4)
    assert opt.at_bound is True
    assert np.allclose([opt.natural["a"], opt.natural["b"]], [10.0, 10.0], atol=1e-3)


def test_optimum_minimize_finds_interior_minimum():
    # convex surface with an interior minimum at x_s = -1/2 B^-1 b = [-0.25, -0.25]
    result = _fit(lambda x1, x2: 50.0 + 2.0 * x1 + 2.0 * x2 + 4.0 * x1**2 + 4.0 * x2**2)
    opt = optimum(result, maximize=False)
    sp = stationary_point(result)
    assert sp.kind == "minimum"
    assert np.allclose(sp.coded, [-0.25, -0.25])
    assert np.allclose(opt.coded, sp.coded, atol=1e-4)
    assert opt.at_bound is False


def test_optimum_respects_custom_bounds():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    # squeeze the box so the maximum clamps to a tighter edge
    opt = optimum(result, maximize=True, bounds=(-0.5, 0.5))
    assert np.allclose(opt.coded, [0.5, 0.5], atol=1e-4)


# --------------------------------------------------------------------------- #
# Natural-unit bounds ({factor_name: (low, high)})
# --------------------------------------------------------------------------- #


def test_optimum_natural_bounds_caps_factor_at_bound():
    # unconstrained interior maximum is at natural a = 5 + 5*(16/47) ~= 6.70 (see
    # test_stationary_point_recovers_interior_maximum); capping a below that forces the
    # constrained optimum onto the cap.
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    opt = optimum(result, maximize=True, bounds={"a": (0.0, 6.0)})

    assert np.isclose(opt.natural["a"], 6.0, atol=1e-3)
    assert opt.at_bound is True
    # b was not bounded and re-optimizes given the clamped a (not the unconstrained 13/47):
    # d/dx2 (2 - 6*x2 - x1) = 0 at x1 = 0.2 -> x2 = 0.3, safely off both b bounds.
    assert np.isclose(opt.coded[1], 0.3, atol=1e-3)


def test_optimum_partial_natural_bounds_defaults_unlisted_factor():
    # only "a" is named, with a loose (non-binding) natural range; "b" must default to the
    # full coded [-1, 1] box and both factors should land at their interior stationary point.
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    sp = stationary_point(result)
    opt = optimum(result, maximize=True, bounds={"a": (0.0, 10.0)})

    assert np.allclose(opt.coded, sp.coded, atol=1e-4)
    assert opt.at_bound is False


def test_optimum_natural_bounds_equivalent_to_coded_tuple():
    # a, b both span natural [0, 10] (center 5, half-range 5), so coded (-0.5, 0.5) is
    # exactly natural (2.5, 7.5) on each factor.
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    opt_coded = optimum(result, maximize=True, bounds=(-0.5, 0.5))
    opt_natural = optimum(result, maximize=True, bounds={"a": (2.5, 7.5), "b": (2.5, 7.5)})

    assert np.allclose(opt_coded.coded, opt_natural.coded, atol=1e-6)
    assert opt_coded.natural == pytest.approx(opt_natural.natural)


def test_optimum_natural_bounds_rejects_unknown_factor():
    result = _fit(lambda x1, x2: 50.0 + 3.0 * x1, model=None, order=1)
    with pytest.raises(ValueError, match="unknown factor"):
        optimum(result, bounds={"c": (0.0, 1.0)})


def test_optimum_natural_bounds_rejects_invalid_range():
    result = _fit(lambda x1, x2: 50.0 + 3.0 * x1, model=None, order=1)
    with pytest.raises(ValueError, match="low < high"):
        optimum(result, bounds={"a": (6.0, 6.0)})


# --------------------------------------------------------------------------- #
# Multi-response desirability (Derringer-Suich)
# --------------------------------------------------------------------------- #


def test_desirability_single_response_maximizes():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1, model=None, order=1)
    goal = ResponseGoal(result, goal="max", low=50.0, high=60.0)
    des = desirability([goal])

    # max response (60) sits at coded x1 = 1, giving full desirability
    assert np.isclose(des.coded[0], 1.0, atol=1e-2)
    assert np.isclose(des.overall, 1.0, atol=1e-3)
    assert np.isclose(des.responses.iloc[0], 60.0, atol=1e-2)


def test_desirability_balances_conflicting_responses():
    # y1 and y2 are identical surfaces but pulled in opposite directions:
    # maximizing y1 and minimizing y2 trade off at coded x1 = 0.5, D = 0.5.
    design = _ccd()
    coded = design.coded().to_numpy()
    y = 50.0 + 10.0 * coded[:, 0]
    r1 = fit_ols(design, y, order=1, interactions=True)
    r2 = fit_ols(design, y, order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=50.0, high=60.0),
        ]
    )
    assert np.isclose(des.coded[0], 0.5, atol=2e-2)
    assert np.isclose(des.overall, 0.5, atol=2e-2)
    assert 0.0 <= des.overall <= 1.0
    assert des.individual.shape == (2,)


def test_desirability_natural_bounds():
    # a spans natural [0, 10] (center 5, half-range 5); restricting it to [5, 10] natural is
    # coded [0, 1], and the response is monotonic increasing in x1, so the optimum sits at the
    # upper end of that restricted range rather than the unconstrained coded x1 = 1.
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1, model=None, order=1)
    goal = ResponseGoal(result, goal="max", low=50.0, high=60.0)
    des = desirability([goal], bounds={"a": (5.0, 10.0)})

    assert np.isclose(des.coded[0], 1.0, atol=1e-2)
    assert np.isclose(des.natural["a"], 10.0, atol=1e-2)
    assert np.isclose(des.responses.iloc[0], 60.0, atol=1e-2)


def test_desirability_requires_matching_factors():
    r1 = _fit(lambda x1, x2: 50.0 + x1, model=None, order=1)
    other = central_composite([ContinuousFactor("p", 0.0, 1.0), ContinuousFactor("q", 0.0, 1.0)])
    r2 = fit_ols(other, np.zeros(other.n_runs), order=1, interactions=True)
    with pytest.raises(ValueError):
        desirability([ResponseGoal(r1, "max", 0.0, 1.0), ResponseGoal(r2, "max", 0.0, 1.0)])


def test_desirability_rejects_same_names_different_bounds():
    # same factor names but different natural bounds -> a shared coded box would decode to
    # different settings per response, so the goals must be rejected (not just name-matched)
    r1 = _fit(lambda x1, x2: 50.0 + x1, model=None, order=1)
    other = central_composite([ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 20.0)])
    r2 = fit_ols(other, np.zeros(other.n_runs), order=1, interactions=True)
    with pytest.raises(ValueError, match="same factors"):
        desirability([ResponseGoal(r1, "max", 0.0, 1.0), ResponseGoal(r2, "max", 0.0, 1.0)])


# --------------------------------------------------------------------------- #
# FitResult convenience methods + readable reprs
# --------------------------------------------------------------------------- #


def test_fitresult_optimum_method_matches_function():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    via_method = result.optimum(maximize=True)
    via_function = optimum(result, maximize=True)
    assert np.allclose(via_method.coded, via_function.coded)
    assert np.isclose(via_method.response, via_function.response)


def test_fitresult_optimum_method_forwards_bounds_and_direction():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    opt = result.optimum(maximize=True, bounds=(-0.5, 0.5))
    assert np.allclose(opt.coded, [0.5, 0.5], atol=1e-4)


def test_fitresult_stationary_point_method_matches_function():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    assert np.allclose(result.stationary_point().coded, stationary_point(result).coded)


def test_optimum_repr_is_readable():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    text = repr(result.optimum(maximize=True))
    assert text.startswith("Optimum(max:")
    assert "a=" in text and "b=" in text
    assert "->" in text
    assert "at bound" in text  # this surface's max clamps to the box edge


def test_stationary_point_repr_reports_kind():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2
    )
    assert repr(stationary_point(result)).startswith("StationaryPoint(maximum:")


def test_desirability_repr_lists_responses():
    # array-fitted goal -> no response_name on the FitResult -> falls back to "response_1"
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1, model=None, order=1)
    text = repr(desirability([ResponseGoal(result, goal="max", low=50.0, high=60.0)]))
    assert text.startswith("DesirabilityResult(D=")
    assert "response_1=" in text


# --------------------------------------------------------------------------- #
# Named responses (FitResult.response_name flowing through to reprs/to_frame)
# --------------------------------------------------------------------------- #


def test_optimum_response_name_from_fit():
    result = _fit(
        lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2, response_name="yield_pct"
    )
    opt = optimum(result, maximize=True)
    assert opt.response_name == "yield_pct"
    assert "yield_pct=" in repr(opt)


def test_optimum_response_name_none_for_array_fit():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2, model=None, order=1)
    opt = optimum(result, maximize=True)
    assert opt.response_name is None
    assert "yield_pct=" not in repr(opt)
    assert "->" in repr(opt)  # bare form retained when no name is known


def test_stationary_point_response_name_from_fit():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2,
        response_name="yield_pct",
    )
    sp = stationary_point(result)
    assert sp.response_name == "yield_pct"
    assert "yield_pct=" in repr(sp)


def test_desirability_response_names_from_fits():
    design = _ccd()
    coded = design.coded().to_numpy()
    yield_design = design.with_response("yield_pct", 50.0 + 10.0 * coded[:, 0])
    impurity_design = design.with_response("impurity_pct", 50.0 - 10.0 * coded[:, 0])
    r1 = fit_ols(yield_design, "yield_pct", order=1, interactions=True)
    r2 = fit_ols(impurity_design, "impurity_pct", order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=40.0, high=50.0),
        ]
    )

    assert isinstance(des.responses, pd.Series)
    assert isinstance(des.individual, pd.Series)
    assert list(des.responses.index) == ["yield_pct", "impurity_pct"]
    assert list(des.individual.index) == ["yield_pct", "impurity_pct"]
    # numeric values still checkable the same way as before (Series supports np.allclose/isclose)
    assert np.isclose(des.responses["yield_pct"], 60.0, atol=1e-2)
    assert "yield_pct=" in repr(des) and "impurity_pct=" in repr(des)


def test_desirability_fallback_labels_for_array_fits():
    design = _ccd()
    coded = design.coded().to_numpy()
    r1 = fit_ols(design, 50.0 + 10.0 * coded[:, 0], order=1, interactions=True)
    r2 = fit_ols(design, 50.0 - 10.0 * coded[:, 0], order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=40.0, high=50.0),
        ]
    )
    assert list(des.responses.index) == ["response_1", "response_2"]
    assert list(des.individual.index) == ["response_1", "response_2"]


def test_desirability_mixed_named_and_fallback_labels():
    design = _ccd()
    coded = design.coded().to_numpy()
    named_design = design.with_response("yield_pct", 50.0 + 10.0 * coded[:, 0])
    r1 = fit_ols(named_design, "yield_pct", order=1, interactions=True)
    r2 = fit_ols(design, 50.0 - 10.0 * coded[:, 0], order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=40.0, high=50.0),
        ]
    )
    assert list(des.responses.index) == ["yield_pct", "response_2"]


# --------------------------------------------------------------------------- #
# to_frame()
# --------------------------------------------------------------------------- #


def test_optimum_to_frame():
    result = _fit(
        lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2, response_name="yield_pct"
    )
    opt = optimum(result, maximize=True)
    frame = opt.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, 3)
    assert list(frame.columns) == ["a", "b", "yield_pct"]
    assert frame.loc[0, "a"] == pytest.approx(opt.natural["a"])
    assert frame.loc[0, "b"] == pytest.approx(opt.natural["b"])
    assert frame.loc[0, "yield_pct"] == pytest.approx(opt.response)


def test_optimum_to_frame_fallback_column_name():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2, model=None, order=1)
    opt = optimum(result, maximize=True)
    assert list(opt.to_frame().columns) == ["a", "b", "predicted"]


def test_stationary_point_to_frame():
    result = _fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2,
        response_name="yield_pct",
    )
    sp = stationary_point(result)
    frame = sp.to_frame()

    assert frame.shape == (1, 3)
    assert list(frame.columns) == ["a", "b", "yield_pct"]
    assert frame.loc[0, "yield_pct"] == pytest.approx(sp.response)


def test_desirability_to_frame():
    design = _ccd()
    coded = design.coded().to_numpy()
    yield_design = design.with_response("yield_pct", 50.0 + 10.0 * coded[:, 0])
    impurity_design = design.with_response("impurity_pct", 50.0 - 10.0 * coded[:, 0])
    r1 = fit_ols(yield_design, "yield_pct", order=1, interactions=True)
    r2 = fit_ols(impurity_design, "impurity_pct", order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=40.0, high=50.0),
        ]
    )
    frame = des.to_frame()

    assert frame.shape == (1, 5)
    assert list(frame.columns) == ["a", "b", "yield_pct", "impurity_pct", "overall_D"]
    assert frame.loc[0, "yield_pct"] == pytest.approx(des.responses["yield_pct"])
    assert frame.loc[0, "overall_D"] == pytest.approx(des.overall)
