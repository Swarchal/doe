"""Phase 2b: response-surface optimization.

Written test-first to document the intended API and the numbers behind each result.
The anchors are quadratics built in coded units on a face-centered CCD (which supports
the full second-order model), so the fit recovers the coefficients exactly and the
stationary-point / optimum maths is checkable against closed-form values.
"""

import numpy as np
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


def _fit(coded_response, *, model="quadratic", order=2, interactions=True):
    """Fit a response defined as a function of the coded factor columns (x1, x2)."""
    design = _ccd()
    coded = design.coded().to_numpy()
    y = coded_response(coded[:, 0], coded[:, 1])
    if model is None:
        return fit_ols(design, y, order=order, interactions=interactions)
    return fit_ols(design, y, model=model)


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
# Multi-response desirability (Derringer-Suich)
# --------------------------------------------------------------------------- #


def test_desirability_single_response_maximizes():
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1, model=None, order=1)
    goal = ResponseGoal(result, goal="max", low=50.0, high=60.0)
    des = desirability([goal])

    # max response (60) sits at coded x1 = 1, giving full desirability
    assert np.isclose(des.coded[0], 1.0, atol=1e-2)
    assert np.isclose(des.overall, 1.0, atol=1e-3)
    assert np.isclose(des.responses[0], 60.0, atol=1e-2)


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


def test_desirability_requires_matching_factors():
    r1 = _fit(lambda x1, x2: 50.0 + x1, model=None, order=1)
    other = central_composite([ContinuousFactor("p", 0.0, 1.0), ContinuousFactor("q", 0.0, 1.0)])
    r2 = fit_ols(other, np.zeros(other.n_runs), order=1, interactions=True)
    with pytest.raises(ValueError):
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
    result = _fit(lambda x1, x2: 50.0 + 10.0 * x1, model=None, order=1)
    text = repr(desirability([ResponseGoal(result, goal="max", low=50.0, high=60.0)]))
    assert text.startswith("DesirabilityResult(D=")
    assert "responses=[" in text
