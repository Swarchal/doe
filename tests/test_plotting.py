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
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import (
    fractional_factorial,
    full_factorial,
    plackett_burman,
)
from doe.generators.rsm import central_composite
from doe.plotting import (
    alias_matrix,
    contour_plot,
    correlation_heatmap,
    half_normal_plot,
    interaction_lines,
    interaction_plot,
    leverage_plot,
    normal_qq,
    predicted_vs_actual,
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
# interaction_lines / interaction_plot
# --------------------------------------------------------------------------- #


def test_interaction_lines_recovers_known_slices():
    # surface: y = 50 + 3 x1 - 2 x2 + 4 x1^2 + 2 x2^2 - x1 x2 (coded). Slicing along a (x1)
    # at the two extremes of b (x2 = -1, +1) gives two analytic quadratics in x1.
    result = _fit_exact()
    nat_x, lines = interaction_lines(result, "a", "b", resolution=21)

    assert nat_x.shape == (21,)
    assert np.isclose(nat_x.min(), 0.0) and np.isclose(nat_x.max(), 10.0)
    # default trace levels are b's low and high
    assert [level for level, _ in lines] == [0.0, 10.0]

    mid = int(np.argmin(np.abs(nat_x - 5.0)))  # a = 5 -> coded x1 = 0
    (b_low, z_low), (b_high, z_high) = lines
    # x1 = 0: b=0 (x2=-1) -> 50+2+2 = 54;  b=10 (x2=+1) -> 50-2+2 = 50
    assert np.isclose(z_low[mid], 54.0, atol=1e-6)
    assert np.isclose(z_high[mid], 50.0, atol=1e-6)


def test_interaction_lines_nonparallel_when_interaction_present():
    # the -x1 x2 term makes the a-slices fan out: the gap between the two lines is not constant
    result = _fit_exact()
    _, lines = interaction_lines(result, "a", "b", resolution=21)
    gap = lines[0][1] - lines[1][1]
    assert not np.allclose(gap, gap[0])


def test_interaction_lines_custom_trace_levels():
    result = _fit_exact()
    nat_x, lines = interaction_lines(result, "a", "b", trace_levels=[2.5, 5.0, 7.5])
    assert [level for level, _ in lines] == [2.5, 5.0, 7.5]
    assert all(z.shape == nat_x.shape for _, z in lines)


def test_interaction_lines_rejects_duplicate_axes():
    result = _fit_exact()
    with pytest.raises(ValueError):
        interaction_lines(result, "a", "a")


def test_interaction_plot_returns_axes_with_legend():
    result = _fit_exact()
    ax = interaction_plot(result, "a", "b")

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "a"
    assert ax.get_ylabel() == "fitted response"
    assert ax.get_legend().get_title().get_text() == "b"
    # one drawn line per trace level (default: 2)
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_interaction_plot_uses_given_axes():
    result = _fit_exact()
    fig, ax = plt.subplots()
    out = interaction_plot(result, "a", "b", ax=ax)
    assert out is ax
    plt.close(fig)


# --------------------------------------------------------------------------- #
# mixed continuous/categorical fits -- plotting over the continuous axes must
# not crash on the categorical contrast columns (regression: KeyError 'cat[B]')
# --------------------------------------------------------------------------- #


def _mixed_fit():
    """A 2^3 fit over two continuous factors + one 2-level categorical, with a clean
    categorical main effect (cat=B is +3 over the mean, cat=A is -3) and no interaction."""
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        ContinuousFactor("time", 0.0, 10.0),
        CategoricalFactor("cat", ("A", "B")),
    ]
    design = full_factorial(factors, levels=2)
    coded = design.coded()
    temp = coded["temp"].to_numpy(dtype=float)
    cat_sign = np.where(design.runs["cat"].to_numpy() == "B", 1.0, -1.0)
    y = 10.0 + 2.0 * temp + 3.0 * cat_sign
    return fit_ols(design, y, order=1, interactions=True)


def test_surface_grid_handles_categorical_third_factor():
    result = _mixed_fit()
    # 'cat[B]' is in the model; sweeping the two continuous factors must not KeyError
    _, _, z_default = surface_grid(result, "temp", "time", resolution=11)
    _, _, z_a = surface_grid(result, "temp", "time", fixed={"cat": "A"}, resolution=11)
    _, _, z_b = surface_grid(result, "temp", "time", fixed={"cat": "B"}, resolution=11)

    assert z_default.shape == (11, 11)
    # default holds the categorical at its average over levels -> midpoint of the two slices
    assert np.allclose(z_default, 0.5 * (z_a + z_b))
    # the categorical main effect (coef ~ 3) is a constant +/-1 contrast swing: z_b - z_a = 6
    assert np.allclose(z_b - z_a, 6.0, atol=1e-6)


def test_surface_grid_rejects_categorical_axis():
    result = _mixed_fit()
    with pytest.raises(TypeError):
        surface_grid(result, "cat", "temp")


def test_surface_grid_rejects_unknown_categorical_level():
    result = _mixed_fit()
    with pytest.raises(ValueError):
        surface_grid(result, "temp", "time", fixed={"cat": "Z"})


def test_interaction_lines_handles_categorical_third_factor():
    result = _mixed_fit()
    nat_x, lines = interaction_lines(result, "temp", "time", resolution=11)
    assert len(lines) == 2
    assert all(np.all(np.isfinite(z)) for _, z in lines)
    # pinning the categorical to B raises every line by the +3 contrast vs the A=-3 slice
    _, lines_a = interaction_lines(result, "temp", "time", fixed={"cat": "A"}, resolution=11)
    _, lines_b = interaction_lines(result, "temp", "time", fixed={"cat": "B"}, resolution=11)
    for (_, za), (_, zb) in zip(lines_a, lines_b, strict=True):
        assert np.allclose(zb - za, 6.0, atol=1e-6)


def test_interaction_lines_rejects_categorical_axis():
    result = _mixed_fit()
    with pytest.raises(TypeError):
        interaction_lines(result, "cat", "temp")
    with pytest.raises(TypeError):
        interaction_lines(result, "temp", "cat")


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


def test_predicted_vs_actual_scatters_observed_against_fitted():
    result = _fit_noisy()
    ax = predicted_vs_actual(result)

    assert isinstance(ax, Axes)
    assert ax.get_xlabel().lower().startswith("actual")
    assert ax.get_ylabel().lower().startswith("predicted")

    observed = result.fitted + result.residuals
    pts = ax.collections[0].get_offsets()
    assert pts.shape[0] == result.fitted.shape[0]
    assert np.allclose(np.sort(pts[:, 0]), np.sort(observed))
    assert np.allclose(np.sort(pts[:, 1]), np.sort(result.fitted))

    # a 45-degree reference line (x-data equals y-data)
    assert any(np.allclose(line.get_xdata(), line.get_ydata()) for line in ax.lines)
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


# --------------------------------------------------------------------------- #
# alias_matrix / correlation_heatmap -- design alias structure
# --------------------------------------------------------------------------- #


def _factors(n):
    import string

    return [ContinuousFactor(c, 0.0, 10.0) for c in string.ascii_lowercase[:n]]


def test_alias_matrix_orthogonal_for_full_factorial():
    # a 2^3 full factorial: every main effect and 2-factor interaction is orthogonal
    labels, corr = alias_matrix(full_factorial(_factors(3), levels=2), interactions=True)

    assert "a:b" in labels  # interactions are included as terms
    assert np.allclose(np.diag(corr), 1.0)
    off_diagonal = corr - np.eye(len(labels))
    assert np.allclose(off_diagonal, 0.0, atol=1e-9)


def test_alias_matrix_detects_confounded_interactions_in_fraction():
    # 2^(4-1) with D=ABC: the classic 2FI aliases are AB=CD, AC=BD, AD=BC
    design = fractional_factorial(_factors(4), generators=["D=ABC"])
    labels, corr = alias_matrix(design, interactions=True)
    idx = {name: i for i, name in enumerate(labels)}

    for left, right in [("a:b", "c:d"), ("a:c", "b:d"), ("a:d", "b:c")]:
        assert np.isclose(abs(corr[idx[left], idx[right]]), 1.0, atol=1e-9)


def test_alias_matrix_partial_aliasing_in_plackett_burman():
    # 8 factors -> a 12-run PB (order-12 base), whose hallmark is *partial* (~1/3) aliasing
    design = plackett_burman(_factors(8))

    # main effects on their own are perfectly orthogonal (identity matrix)
    labels, corr = alias_matrix(design, interactions=False)
    assert np.allclose(corr - np.eye(len(labels)), 0.0, atol=1e-9)

    # but with interactions, mains leak *partially* into 2FIs: magnitudes strictly
    # between 0 and 1 appear (PB's signature complex aliasing, ~1/3)
    labels2, corr2 = alias_matrix(design, interactions=True, absolute=True)
    off = corr2[~np.eye(len(labels2), dtype=bool)]
    assert np.all(off >= -1e-9)  # absolute=True -> non-negative
    assert np.any((off > 0.1) & (off < 0.99))


def test_correlation_heatmap_returns_axes_matching_alias_matrix():
    design = fractional_factorial(_factors(4), generators=["D=ABC"])
    ax = correlation_heatmap(design)

    assert isinstance(ax, Axes)
    labels, corr = alias_matrix(design)
    # the displayed image is the alias matrix itself
    assert ax.images[0].get_array().shape == (len(labels), len(labels))
    assert [t.get_text() for t in ax.get_xticklabels()] == labels
    plt.close(ax.figure)


def test_correlation_heatmap_uses_given_axes():
    fig, ax = plt.subplots()
    out = correlation_heatmap(full_factorial(_factors(2), levels=2), ax=ax)
    assert out is ax
    plt.close(fig)


def test_leverage_plot_accepts_design():
    design = full_factorial(_factors(2), levels=2)
    ax = leverage_plot(design)

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Run"
    assert ax.get_ylabel() == "Leverage"
    assert ax.lines[0].get_xdata().shape == (design.n_runs,)
    # reference line is the 2p/n rule, after the run-leverage line
    assert np.allclose(ax.lines[1].get_ydata(), 2.0)
    plt.close(ax.figure)


def test_leverage_plot_accepts_fit_result():
    result = _fit_exact()
    ax = leverage_plot(result)

    assert isinstance(ax, Axes)
    assert ax.lines[0].get_xdata().shape == (result.model_matrix.shape[0],)
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# Phase 4b: ternary (mixture) plots
# --------------------------------------------------------------------------- #


def _yarn_fit():
    """A Scheffé-quadratic fit over Cornell's yarn components (exact synthetic response)."""
    from doe.factors import MixtureFactor
    from doe.generators.mixture import simplex_centroid

    components = [MixtureFactor("a"), MixtureFactor("b"), MixtureFactor("c")]
    design = simplex_centroid(components)
    x = design.runs.to_numpy(dtype=float)
    # known blending surface: y = 2a + 3b + 4c + 6ab
    y = 2 * x[:, 0] + 3 * x[:, 1] + 4 * x[:, 2] + 6 * x[:, 0] * x[:, 1]
    return fit_ols(design, y, model="scheffe-quadratic"), design


def test_ternary_grid_predicts_known_blending_surface():
    from doe.plotting import ternary_grid

    result, _design = _yarn_fit()
    x, y, z, points = ternary_grid(result, resolution=10)

    assert x.shape == y.shape == z.shape == (points.shape[0],)
    assert np.allclose(points.sum(axis=1), 1.0, atol=1e-12)
    expected = (
        2 * points[:, 0] + 3 * points[:, 1] + 4 * points[:, 2]
        + 6 * points[:, 0] * points[:, 1]
    )
    assert np.allclose(z, expected, atol=1e-8)
    # Cartesian mapping puts the three pure blends at the triangle's corners
    assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(1.0)
    assert y.min() == pytest.approx(0.0)
    assert y.max() == pytest.approx(np.sqrt(3.0) / 2.0)


def test_ternary_grid_requires_three_mixture_components():
    from doe.plotting import ternary_grid

    result = _fit_exact()  # a non-mixture fit
    with pytest.raises(ValueError, match="3 mixture components"):
        ternary_grid(result)


def test_ternary_contour_draws_and_overlays_design_points():
    from doe.plotting import ternary_contour

    result, design = _yarn_fit()
    ax = ternary_contour(result, design, resolution=15)

    assert isinstance(ax, Axes)
    # the design overlay is present with one marker per run
    offsets = ax.collections[-1].get_offsets()
    assert offsets.shape[0] == design.n_runs
    plt.close(ax.figure)
