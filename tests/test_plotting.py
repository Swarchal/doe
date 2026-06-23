"""Phase 2a: response-surface contour + regression-diagnostic plots.

Written test-first to document the intended API and the numbers behind each plot.
The plotting helpers lazy-import matplotlib; ``tests/conftest.py`` forces the Agg
backend so these run headless.

Design decisions these tests pin down:

* ``surface_grid`` is the headless, numerically-testable core: it evaluates the fitted
  surface over a grid of two chosen factors (others held fixed) and returns natural-unit
  ``(X, Y, Z)`` mesh arrays. ``contour_plot`` is a thin matplotlib wrapper over it.
* Contours are drawn in **natural units**, which means ``FitResult`` must know its factors
  (a ``factors`` field is added so a fitted result is self-describing in natural units).
* ``fixed`` holds the remaining factors at given **natural** values (default: factor center).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from doe.analysis.fit import fit_ols
from doe.factors import ContinuousFactor
from doe.generators.rsm import central_composite
from doe.plotting import (
    contour_plot,
    half_normal_plot,
    normal_qq,
    residuals_vs_fitted,
    surface_grid,
    surface_plot,
)


def _ccd(center=5):
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    return central_composite(factors, center=center)


def _ccd3(center=6):
    factors = [
        ContinuousFactor("a", 0.0, 10.0),
        ContinuousFactor("b", 0.0, 10.0),
        ContinuousFactor("c", 0.0, 10.0),
    ]
    return central_composite(factors, center=center)


def _surface(design):
    """y = 50 + 3*x1 - 2*x2 + 4*x1^2 + 2*x2^2 - 1*x1*x2  (coded units, both squares present)."""
    coded = design.coded().to_numpy()
    x1, x2 = coded[:, 0], coded[:, 1]
    return 50.0 + 3.0 * x1 - 2.0 * x2 + 4.0 * x1**2 + 2.0 * x2**2 - 1.0 * x1 * x2


def _fit_exact():
    design = _ccd()
    return fit_ols(design, _surface(design), model="quadratic")


def _fit_noisy(seed=0, scale=0.5):
    design = _ccd()
    rng = np.random.default_rng(seed)
    y = _surface(design) + rng.normal(scale=scale, size=design.n_runs)
    return fit_ols(design, y, model="quadratic")


# --------------------------------------------------------------------------- #
# surface_grid -- the headless surface evaluator behind contour_plot
# --------------------------------------------------------------------------- #


def test_surface_grid_shapes_and_natural_extent():
    result = _fit_exact()
    x_grid, y_grid, z_grid = surface_grid(result, "a", "b", resolution=21)

    assert x_grid.shape == (21, 21)
    assert y_grid.shape == (21, 21)
    assert z_grid.shape == (21, 21)
    # the grid spans the factors' natural bounds
    assert np.isclose(x_grid.min(), 0.0) and np.isclose(x_grid.max(), 10.0)
    assert np.isclose(y_grid.min(), 0.0) and np.isclose(y_grid.max(), 10.0)


def test_surface_grid_recovers_known_quadratic():
    result = _fit_exact()
    x_grid, y_grid, z_grid = surface_grid(result, "a", "b", resolution=21)

    def value_at(x_nat, y_nat):
        j = int(np.argmin(np.abs(x_grid[0, :] - x_nat)))
        i = int(np.argmin(np.abs(y_grid[:, 0] - y_nat)))
        return z_grid[i, j]

    # center (5,5) -> coded (0,0) -> intercept = 50
    assert np.isclose(value_at(5.0, 5.0), 50.0, atol=1e-6)
    # corner (10,10) -> coded (1,1): 50+3-2+4+2-1 = 56
    assert np.isclose(value_at(10.0, 10.0), 56.0, atol=1e-6)
    # corner (0,0) -> coded (-1,-1): 50-3+2+4+2-1 = 54
    assert np.isclose(value_at(0.0, 0.0), 54.0, atol=1e-6)


def test_surface_grid_holds_other_factors_fixed():
    design = _ccd3()
    coded = design.coded().to_numpy()
    x1, x3 = coded[:, 0], coded[:, 2]
    y = 10.0 + 2.0 * x1 + 3.0 * x3  # depends on c (the 3rd factor)
    result = fit_ols(design, y, model="quadratic")

    # default: omitted factor c held at its center (coded 0) -> no contribution
    _, _, z_center = surface_grid(result, "a", "b", resolution=11)
    # hold c at natural 10.0 (coded +1) -> a constant +3 everywhere on the a-b surface
    _, _, z_high = surface_grid(result, "a", "b", fixed={"c": 10.0}, resolution=11)

    assert np.allclose(z_high - z_center, 3.0, atol=1e-6)


def test_surface_grid_rejects_duplicate_axes():
    result = _fit_exact()
    with pytest.raises(ValueError):
        surface_grid(result, "a", "a")


def test_surface_grid_rejects_unknown_factor():
    result = _fit_exact()
    with pytest.raises((KeyError, ValueError)):
        surface_grid(result, "a", "does_not_exist")


# --------------------------------------------------------------------------- #
# contour_plot
# --------------------------------------------------------------------------- #


def test_contour_plot_returns_axes_with_natural_labels():
    result = _fit_exact()
    ax = contour_plot(result, "a", "b")

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "a"
    assert ax.get_ylabel() == "b"
    # the filled contour sets the data limits to the natural factor ranges
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    assert xlo <= 1e-9 and xhi >= 10.0 - 1e-9
    assert ylo <= 1e-9 and yhi >= 10.0 - 1e-9
    plt.close(ax.figure)


def test_contour_plot_uses_given_axes():
    result = _fit_exact()
    fig, ax = plt.subplots()
    out = contour_plot(result, "a", "b", ax=ax)
    assert out is ax
    plt.close(fig)


# --------------------------------------------------------------------------- #
# surface_plot (Phase 2b 3-D companion)
# --------------------------------------------------------------------------- #


def test_surface_plot_returns_3d_axes_with_natural_labels():
    result = _fit_exact()
    ax = surface_plot(result, "a", "b", resolution=11)

    assert ax.name == "3d"  # a 3-D projection axes
    assert ax.get_xlabel() == "a"
    assert ax.get_ylabel() == "b"
    assert ax.get_zlabel() == "fitted response"
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# residual diagnostics
# --------------------------------------------------------------------------- #


def test_residuals_vs_fitted_plots_fitted_against_residuals():
    result = _fit_noisy()
    ax = residuals_vs_fitted(result)

    assert isinstance(ax, Axes)
    assert ax.get_xlabel().lower().startswith("fitted")
    assert "resid" in ax.get_ylabel().lower()

    pts = ax.collections[0].get_offsets()
    assert pts.shape[0] == result.fitted.shape[0]
    assert np.allclose(np.sort(pts[:, 0]), np.sort(result.fitted))
    assert np.allclose(np.sort(pts[:, 1]), np.sort(result.residuals))

    # a horizontal reference line at zero
    assert any(np.allclose(line.get_ydata(), 0.0) for line in ax.lines)
    plt.close(ax.figure)


def test_normal_qq_draws_points_and_reference_line():
    result = _fit_noisy()
    ax = normal_qq(result)

    assert isinstance(ax, Axes)
    # scipy.stats.probplot draws the ordered residuals plus a best-fit line: 2 lines
    assert len(ax.lines) >= 2
    ordered = ax.lines[0].get_ydata()
    assert len(ordered) == result.residuals.shape[0]
    # the points are the residuals in sorted order
    assert np.allclose(np.sort(ordered), np.sort(result.residuals))
    plt.close(ax.figure)


def test_half_normal_plot_one_point_per_effect():
    result = _fit_exact()
    ax = half_normal_plot(result)

    assert isinstance(ax, Axes)
    assert "effect" in ax.get_ylabel().lower()

    pts = ax.collections[0].get_offsets()
    n_effects = len(result.term_names) - 1  # drop the intercept
    assert pts.shape[0] == n_effects
    # y-positions are the absolute effects (intercept excluded)
    assert np.allclose(np.sort(pts[:, 1]), np.sort(np.abs(result.effects[1:])))
    # half-normal quantiles are non-negative and increase with |effect|
    assert np.all(pts[:, 0] >= -1e-9)
    plt.close(ax.figure)
