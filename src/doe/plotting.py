"""Plotting helpers (Phase 1: main-effects and Pareto-of-effects).

Importing this module requires the optional ``matplotlib`` dependency
(``pip install 'doe[plotting]'``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .analysis.diagnostics import correlation_matrix as _correlation_matrix
from .analysis.diagnostics import leverage as _leverage
from .analysis.fit import FitResult
from .analysis.model import _effect_code, build_model_matrix
from .design import Design
from .factors import CategoricalFactor, ContinuousFactor, FactorSet

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def pareto_plot(result: FitResult, ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of absolute effect sizes, largest first (intercept dropped).

    Ranking effects by magnitude makes the "vital few vs. trivial many" pattern obvious: a
    handful of large bars are the factors that drive the response, and the long tail of small
    bars are candidates to pool into error. Sign is dropped because screening asks *which*
    factors matter, not their direction.
    """
    import matplotlib.pyplot as plt

    names = result.term_names[1:]
    effects = np.abs(result.effects[1:])
    order = np.argsort(effects)
    names = [names[i] for i in order]
    effects = effects[order]

    if ax is None:
        _, ax = plt.subplots()
    ax.barh(names, effects)
    ax.set_xlabel("|effect|")
    ax.set_title("Pareto plot of effects")
    return ax


def main_effects_plot(result: FitResult, ax: Axes | None = None) -> Axes:
    """Plot each main-effect coefficient as a point (signed magnitude)."""
    import matplotlib.pyplot as plt

    pairs = [
        (name, coef)
        for name, coef in zip(result.term_names, result.coefficients, strict=True)
        if name != "Intercept" and ":" not in name and "^" not in name
    ]
    names = [p[0] for p in pairs]
    coefs = [p[1] for p in pairs]

    if ax is None:
        _, ax = plt.subplots()
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.plot(names, coefs, "o-")
    ax.set_ylabel("coefficient (coded units)")
    ax.set_title("Main effects")
    return ax


# --------------------------------------------------------------------------- #
# Phase 2a: response-surface + diagnostic plots
# --------------------------------------------------------------------------- #


def _term_column(name: str, coded: dict[str, np.ndarray], sample: np.ndarray) -> np.ndarray:
    """Reconstruct one model-matrix column from its term name and coded factor grids."""
    if name == "Intercept":
        return np.ones_like(sample)
    if ":" in name:  # two-factor interaction "a:b"
        a, b = name.split(":")
        return np.asarray(coded[a] * coded[b], dtype=float)
    if "^" in name:  # quadratic term "a^2"
        base, power = name.split("^")
        return np.asarray(coded[base] ** int(power), dtype=float)
    return coded[name]


def _predict(result: FitResult, coded: dict[str, np.ndarray], sample: np.ndarray) -> np.ndarray:
    """Evaluate the fitted model on coded factor grids, matching ``result.term_names``."""
    total = np.zeros_like(sample, dtype=float)
    for name, coef in zip(result.term_names, result.coefficients, strict=True):
        total = total + coef * _term_column(name, coded, sample)
    return np.asarray(total, dtype=float)


def _require_continuous_axes(fs: FactorSet, axes: Sequence[str]) -> None:
    """Validate that each named plot axis exists and is continuous.

    Plot axes are swept over the coded ``[-1, +1]`` grid, which only makes sense for a
    continuous factor; a categorical factor has no such ordered scale. Raises ``KeyError``
    for an unknown name and ``TypeError`` for a categorical one (a clearer signal than the
    ``AttributeError`` that ``code``/``decode`` would otherwise raise downstream).
    """
    for axis in axes:
        if not isinstance(fs[axis], ContinuousFactor):  # ``fs[axis]`` raises KeyError if unknown
            raise TypeError(
                f"factor {axis!r} is categorical; only continuous factors can be a plot axis"
            )


def _coded_columns(
    result: FitResult,
    sweeps: dict[str, np.ndarray],
    fixed: Mapping[str, object],
    sample: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the ``{model-column name: grid}`` mapping that :func:`_predict` consumes.

    ``sweeps`` holds the coded grids for the (continuous) factors being varied; ``fixed`` holds
    remaining factors at given *natural* values. Any factor neither swept nor fixed is held at
    its center -- coded ``0`` for a continuous factor, and *all contrasts ``0``* for a
    categorical one, which is its average over levels (the marginal surface). Categorical
    factors are expanded into their ``factor[level]`` contrast columns via the same
    :func:`~doe.analysis.model._effect_code` used to fit the model, so the keys line up exactly
    with ``result.term_names`` -- including for mixed continuous/categorical fits.
    """
    fs = result.factors
    unknown = set(fixed) - set(fs.names)
    if unknown:
        raise ValueError(f"fixed refers to unknown factors: {sorted(unknown)}")

    coded: dict[str, np.ndarray] = {}
    for factor in fs:
        name = factor.name
        if isinstance(factor, ContinuousFactor):
            if name in sweeps:
                coded[name] = sweeps[name]
            elif name in fixed:
                coded[name] = np.full_like(sample, factor.code(np.asarray(fixed[name])))
            else:
                coded[name] = np.zeros_like(sample)  # held at center
        elif isinstance(factor, CategoricalFactor):
            # categorical -> one constant column per effect-coded contrast
            if name in fixed:
                # _effect_code validates the level and yields the contrast values for it
                encoding = _effect_code(factor, np.asarray([fixed[name]], dtype=object))
                for contrast_name, contrast_col in encoding:
                    coded[contrast_name] = np.full_like(sample, float(contrast_col[0]))
            else:  # marginal: average over levels is all contrasts == 0
                encoding = _effect_code(factor, np.asarray(list(factor.levels), dtype=object))
                for contrast_name, _ in encoding:
                    coded[contrast_name] = np.zeros_like(sample)
    return coded


def surface_grid(
    result: FitResult,
    x: str,
    y: str,
    *,
    fixed: Mapping[str, object] | None = None,
    resolution: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the fitted surface over a grid of two factors (others held fixed).

    Returns natural-unit mesh arrays ``(X, Y, Z)`` of shape ``(resolution, resolution)``,
    with ``X`` varying along the columns and ``Y`` along the rows (``contourf`` convention).
    ``fixed`` holds the remaining factors at given *natural* values (default: factor center; a
    categorical factor defaults to its average over levels). The ``x``/``y`` axes must be
    continuous. This is the headless core that :func:`contour_plot` draws.
    """
    fs = result.factors
    if x == y:
        raise ValueError("x and y must be two different factors")
    _require_continuous_axes(fs, (x, y))

    cx = np.linspace(-1.0, 1.0, resolution)
    cy = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y = np.meshgrid(cx, cy)  # (resolution, resolution)

    coded = _coded_columns(result, {x: grid_x, y: grid_y}, dict(fixed or {}), grid_x)
    z = _predict(result, coded, grid_x)
    nat_x = fs[x].decode(grid_x)  # type: ignore[union-attr]
    nat_y = fs[y].decode(grid_y)  # type: ignore[union-attr]
    return nat_x, nat_y, z


def contour_plot(
    result: FitResult,
    x: str,
    y: str,
    *,
    fixed: Mapping[str, object] | None = None,
    ax: Axes | None = None,
    resolution: int = 25,
    filled: bool = True,
) -> Axes:
    """Filled contour of the fitted surface over two factors, in natural units."""
    import matplotlib.pyplot as plt

    nat_x, nat_y, z = surface_grid(result, x, y, fixed=fixed, resolution=resolution)

    if ax is None:
        _, ax = plt.subplots()
    if filled:
        mappable = ax.contourf(nat_x, nat_y, z, levels=12)
        ax.figure.colorbar(mappable, ax=ax, label="fitted response")
    lines = ax.contour(nat_x, nat_y, z, levels=8, colors="k", linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Fitted response surface")
    return ax


def surface_plot(
    result: FitResult,
    x: str,
    y: str,
    *,
    fixed: Mapping[str, object] | None = None,
    ax: Axes | None = None,
    resolution: int = 25,
    cmap: str = "viridis",
) -> Axes:
    """3-D surface of the fitted response over two factors (Phase 2b companion to contour_plot).

    Reuses :func:`surface_grid` for the natural-unit mesh. Requires matplotlib's ``mplot3d``
    (bundled with matplotlib); passing your own ``ax`` requires a 3-D projection axes.
    """
    import matplotlib.pyplot as plt

    nat_x, nat_y, z = surface_grid(result, x, y, fixed=fixed, resolution=resolution)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    ax.plot_surface(nat_x, nat_y, z, cmap=cmap)  # type: ignore[union-attr]
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel("fitted response")  # type: ignore[union-attr]
    ax.set_title("Fitted response surface")
    return ax


def interaction_lines(
    result: FitResult,
    x: str,
    trace: str,
    *,
    fixed: Mapping[str, object] | None = None,
    trace_levels: Sequence[float] | None = None,
    resolution: int = 25,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """Evaluate the fitted model along ``x`` for several fixed levels of ``trace``.

    Returns ``(nat_x, lines)`` where ``nat_x`` is the natural-unit sweep of factor ``x``
    (``resolution`` points across its range) and ``lines`` is a list of
    ``(trace_natural_value, z)`` pairs, one per ``trace`` level. Any remaining factors are
    held at the value in ``fixed`` (natural units) or at their center. ``trace_levels`` are
    natural values for ``trace`` (default: its low and high).

    This is the headless, numerically-testable core that :func:`interaction_plot` draws. The
    lines are parallel exactly when ``x`` and ``trace`` do not interact; a fanning gap between
    them is the interaction. ``x`` and ``trace`` must be continuous (a categorical factor has
    no coded sweep), matching :func:`surface_grid`; any remaining factor is held per ``fixed``.
    """
    fs = result.factors
    if x == trace:
        raise ValueError("x and trace must be two different factors")
    _require_continuous_axes(fs, (x, trace))

    trace_factor = fs[trace]
    if trace_levels is None:
        trace_levels = [trace_factor.low, trace_factor.high]  # type: ignore[union-attr]

    cx = np.linspace(-1.0, 1.0, resolution)
    nat_x = fs[x].decode(cx)  # type: ignore[union-attr]

    # Build the coded columns once (trace held at center for now); each line only re-sets the
    # one trace column, so the x sweep, fixed factors, and any held factors are not rebuilt.
    coded = _coded_columns(result, {x: cx}, dict(fixed or {}), cx)
    lines: list[tuple[float, np.ndarray]] = []
    for level in trace_levels:
        coded[trace] = np.full_like(cx, trace_factor.code(np.asarray(level)))  # type: ignore[union-attr]
        z = _predict(result, coded, cx)
        lines.append((float(level), z))
    return nat_x, lines


def interaction_plot(
    result: FitResult,
    x: str,
    trace: str,
    *,
    fixed: Mapping[str, object] | None = None,
    trace_levels: Sequence[float] | None = None,
    ax: Axes | None = None,
    resolution: int = 25,
) -> Axes:
    """Plot the fitted response against ``x`` as a separate line per level of ``trace``.

    The classic two-factor interaction plot: parallel lines mean the effect of ``x`` does not
    depend on ``trace`` (no interaction), while lines that fan apart or cross reveal one. Draws
    :func:`interaction_lines`; both factors must be continuous.
    """
    import matplotlib.pyplot as plt

    nat_x, lines = interaction_lines(
        result, x, trace, fixed=fixed, trace_levels=trace_levels, resolution=resolution
    )

    if ax is None:
        _, ax = plt.subplots()
    trace_units = getattr(result.factors[trace], "units", None)
    suffix = f" {trace_units}" if trace_units else ""
    for level, z in lines:
        ax.plot(nat_x, z, "-o", markersize=3, label=f"{level:g}{suffix}")
    ax.set_xlabel(x)
    ax.set_ylabel("fitted response")
    ax.legend(title=trace)
    ax.set_title(f"Interaction: {x} × {trace}")
    return ax


def alias_matrix(
    design: Design,
    *,
    order: int = 1,
    interactions: bool = True,
    absolute: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Pairwise correlations among model terms -- the design's alias structure.

    Returns ``(labels, matrix)`` where ``matrix[i, j]`` is the Pearson correlation between
    model terms ``labels[i]`` and ``labels[j]`` in coded units. The intercept (and any other
    constant column, e.g. a squared pure-``+/-1`` term) is dropped, having no defined
    correlation. Read it as: off-diagonals near ``0`` mean the terms are estimated
    independently (an orthogonal design); ``|r| = 1`` is full aliasing (the terms are
    confounded -- you cannot separate their effects); intermediate magnitudes are *partial*
    aliasing (e.g. a Plackett-Burman main effect leaking ``+/- 1/3`` into two-factor
    interactions). With ``absolute=True`` the magnitudes ``|r|`` are returned instead.

    This is the headless, numerically-testable core that :func:`correlation_heatmap` draws.
    The model whose aliasing is assessed is chosen by ``order``/``interactions``, exactly as
    in :func:`~doe.analysis.model.build_model_matrix`.
    """
    mm = build_model_matrix(design, order=order, interactions=interactions)
    corr_df = _correlation_matrix(mm.X, mm.term_names)
    labels = list(corr_df.index)
    corr = corr_df.to_numpy(copy=True)
    np.clip(corr, -1.0, 1.0, out=corr)  # tame rounding drift outside [-1, 1]
    if absolute:
        corr = np.abs(corr)
    return labels, corr


def correlation_heatmap(
    design: Design,
    *,
    order: int = 1,
    interactions: bool = True,
    absolute: bool = False,
    ax: Axes | None = None,
    cmap: str | None = None,
    annotate: bool | None = None,
) -> Axes:
    """Heatmap of the design's alias structure (term-to-term correlations).

    Draws :func:`alias_matrix`. A clean block-diagonal of zeros off the diagonal means an
    orthogonal design; bright off-diagonal cells flag aliased (confounded) terms. ``annotate``
    overlays the numeric values (default: on when there are <= 12 terms). ``absolute`` shows
    ``|r|`` on a sequential scale -- handy for spotting *any* aliasing regardless of sign.
    """
    import matplotlib.pyplot as plt

    labels, corr = alias_matrix(
        design, order=order, interactions=interactions, absolute=absolute
    )
    n = len(labels)

    if ax is None:
        _, ax = plt.subplots()
    if cmap is None:
        cmap = "magma" if absolute else "RdBu_r"
    vmin = 0.0 if absolute else -1.0
    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)

    if annotate is None:
        annotate = n <= 12
    if annotate:
        for i in range(n):
            for j in range(n):
                mag = abs(corr[i, j])
                # pick legible text colour against each colormap's dark/light extremes
                light_bg = mag >= 0.5 if absolute else mag < 0.5
                ax.text(
                    j,
                    i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if light_bg else "white",
                )

    label = "|correlation|" if absolute else "correlation"
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title("Alias structure (term correlations)")
    return ax


def residuals_vs_fitted(result: FitResult, ax: Axes | None = None) -> Axes:
    """Scatter of residuals against fitted values, with a zero reference line.

    Checks the constant-variance and adequacy assumptions of OLS: a healthy plot is a
    structureless band around zero. A funnel shape signals non-constant variance (consider a
    transform), and curvature signals a missing higher-order term.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.scatter(result.fitted, result.residuals)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    return ax


def leverage_plot(result_or_design: FitResult | Design, ax: Axes | None = None) -> Axes:
    """Leverage (hat-matrix diagonal) per run, with the ``2p/n`` high-leverage reference line.

    The design-evaluation companion to the residual diagnostics: leverage measures how much a
    run's factor settings let it pull on the fit, independent of its response. Runs above the
    ``2p/n`` rule-of-thumb line are high-leverage points whose loss would most degrade the
    design. Accepts a fitted :class:`~doe.analysis.fit.FitResult` (uses its model matrix) or a
    bare :class:`~doe.design.Design` (expands a model matrix first). Draws
    :func:`doe.analysis.diagnostics.leverage`.
    """
    import matplotlib.pyplot as plt

    if isinstance(result_or_design, FitResult):
        x = result_or_design.model_matrix
    elif isinstance(result_or_design, Design):
        x = build_model_matrix(result_or_design).X
    else:
        raise TypeError("leverage_plot expects a FitResult or Design")

    h = _leverage(x)
    n_runs, n_terms = x.shape
    threshold = 2.0 * n_terms / n_runs
    run_numbers = np.arange(1, n_runs + 1)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(run_numbers, h, "o-")
    ax.axhline(threshold, color="grey", linestyle="--", lw=0.8, label="2p/n")
    ax.set_xlabel("Run")
    ax.set_ylabel("Leverage")
    ax.set_title("Leverage by run")
    ax.legend()
    return ax


def predicted_vs_actual(result: FitResult, ax: Axes | None = None) -> Axes:
    """Scatter of predicted (fitted) against observed responses, with a 45° reference line.

    The most direct read on model adequacy: points hug the ``y = x`` line when the model
    reproduces the data and scatter away from it when it does not, with systematic curvature
    about the line flagging a missing term. The observed values are reconstructed as
    ``fitted + residuals`` (so no separate response array is needed), and ``R²`` -- the same
    fraction-of-variance the line's tightness shows -- is reported in the title.
    """
    import matplotlib.pyplot as plt

    observed = result.fitted + result.residuals
    predicted = result.fitted

    if ax is None:
        _, ax = plt.subplots()
    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=0.8, zorder=0)
    ax.scatter(observed, predicted)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Predicted vs actual (R² = {result.r_squared:.3f})")
    return ax


def normal_qq(result: FitResult, ax: Axes | None = None) -> Axes:
    """Normal quantile-quantile plot of the residuals (``scipy.stats.probplot``).

    Checks the normality assumption that the t- and F-tests rely on: residuals that are roughly
    normal fall on the reference line. Heavy tails or marked curvature warn that p-values should
    be read with caution.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if ax is None:
        _, ax = plt.subplots()
    stats.probplot(result.residuals, plot=ax)
    ax.set_title("Normal Q-Q")
    return ax


def half_normal_plot(result: FitResult, ax: Axes | None = None) -> Axes:
    """Absolute effects against half-normal quantiles (screening companion to pareto_plot).

    This is the standard way to judge significance when a saturated design leaves no degrees of
    freedom for a formal test. Inactive effects are just noise, so they scatter along a straight
    line through the origin; genuinely active effects break above that line on the right. The eye
    picks out the real factors as the points that depart from the noise trend.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if ax is None:
        _, ax = plt.subplots()

    # rank effects by magnitude and plot against half-normal quantiles: if the effects were pure
    # noise these points would fall on a line, so departures from it flag the active factors.
    abs_effects = np.abs(result.effects[1:])  # drop the intercept
    names = result.term_names[1:]
    order = np.argsort(abs_effects)
    abs_effects = abs_effects[order]
    names = [names[i] for i in order]

    m = len(abs_effects)
    ranks = (np.arange(1, m + 1) - 0.5) / m
    quantiles = stats.norm.ppf(0.5 + 0.5 * ranks)  # half-normal quantiles (>= 0)

    ax.scatter(quantiles, abs_effects)
    for q, e, name in zip(quantiles, abs_effects, names, strict=True):
        ax.annotate(name, (q, e), textcoords="offset points", xytext=(4, 0), fontsize=8)
    ax.set_xlabel("half-normal quantile")
    ax.set_ylabel("|effect|")
    ax.set_title("Half-normal plot of effects")
    return ax


# --------------------------------------------------------------------------- #
# Phase 4b: mixture (ternary) plots
# --------------------------------------------------------------------------- #


def ternary_grid(
    result: FitResult, *, resolution: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a fitted Scheffé model over the 3-component simplex (headless core).

    Samples the ``1/resolution`` lattice over the full simplex, predicts through the same
    Scheffé expansion the model was fitted with, and maps barycentric proportions
    ``(a, b, c)`` to Cartesian plot coordinates (component vertices at ``(0, 0)``,
    ``(1, 0)``, and ``(0.5, sqrt(3)/2)``). Returns ``(x, y, z, points)`` where ``x``/``y``/
    ``z`` are flat arrays for triangular contouring and ``points`` is the ``(n, 3)`` array
    of proportions. This is the headless core :func:`ternary_contour` draws.
    """
    from .analysis.model import expand_coded_points

    fs = result.factors
    if not fs.is_mixture or len(fs) != 3:
        raise ValueError(
            "ternary_grid requires a fit over exactly 3 mixture components; for more "
            "components, fix all but three and refit, or read the coefficients directly"
        )
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    # the 1/resolution lattice over the whole simplex
    pts = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            pts.append((i / resolution, j / resolution, k / resolution))
    points = np.asarray(pts, dtype=float)

    mm = expand_coded_points(points, fs, order=result.order, interactions=result.interactions)
    z = mm.X @ result.coefficients

    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3.0) / 2.0) * points[:, 2]
    return x, y, np.asarray(z, dtype=float), points


def ternary_contour(
    result: FitResult,
    design: Design | None = None,
    *,
    resolution: int = 100,
    ax: Axes | None = None,
    filled: bool = True,
) -> Axes:
    """Ternary (simplex) contour of a fitted 3-component Scheffé blending model.

    The mixture counterpart of :func:`contour_plot`: the fitted surface is drawn over the
    triangular composition space, with each vertex a pure component. Pass the ``design`` to
    overlay its blends as points. Requires exactly 3 mixture components (see
    :func:`ternary_grid`).
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    x, y, z, _points = ternary_grid(result, resolution=resolution)

    if ax is None:
        _, ax = plt.subplots()
    if filled:
        mappable = ax.tricontourf(x, y, z, levels=12)
        ax.figure.colorbar(mappable, ax=ax, label="fitted response")
    # A white halo keeps the black contour lines and their labels legible over both
    # the dark (low) and bright (high) ends of the filled surface.
    halo: list[pe.AbstractPathEffect] = [pe.withStroke(linewidth=1.6, foreground="white")]
    lines = ax.tricontour(x, y, z, levels=8, colors="k", linewidths=0.8)
    lines.set(path_effects=halo)
    labels = ax.clabel(lines, inline=True, fontsize=8)
    for label in labels:
        label.set_path_effects(halo)

    names = result.factors.names
    height = np.sqrt(3.0) / 2.0
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, height, 0.0]
    ax.plot(triangle_x, triangle_y, color="k", lw=1.0)
    ax.annotate(
        names[0], (0.0, 0.0), textcoords="offset points", xytext=(-8, -12), ha="right"
    )
    ax.annotate(
        names[1], (1.0, 0.0), textcoords="offset points", xytext=(0, -14), ha="right"
    )
    ax.annotate(
        names[2], (0.5, height), textcoords="offset points", xytext=(0, 8), ha="center"
    )

    if design is not None:
        props = design.runs[names].to_numpy(dtype=float)
        px = props[:, 1] + 0.5 * props[:, 2]
        py = height * props[:, 2]
        ax.scatter(
            px, py, marker="o", color="crimson", s=55, zorder=3,
            edgecolors="white", linewidths=1.2, clip_on=False,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()
    # Extra margins + title pad so the apex label clears the title and the
    # bottom-right vertex label clears the colorbar.
    ax.margins(x=0.08, y=0.08)
    ax.set_title("Fitted blending surface", pad=22)
    return ax
