"""Plotting helpers (Phase 1: main-effects and Pareto-of-effects).

Importing this module requires the optional ``matplotlib`` dependency
(``pip install 'doe[plotting]'``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .analysis.fit import FitResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def pareto_plot(result: FitResult, ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of absolute effect sizes, largest first (intercept dropped)."""
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


def surface_grid(
    result: FitResult,
    x: str,
    y: str,
    *,
    fixed: dict[str, float] | None = None,
    resolution: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the fitted surface over a grid of two factors (others held fixed).

    Returns natural-unit mesh arrays ``(X, Y, Z)`` of shape ``(resolution, resolution)``,
    with ``X`` varying along the columns and ``Y`` along the rows (``contourf`` convention).
    ``fixed`` holds the remaining factors at given *natural* values (default: factor center).
    This is the headless core that :func:`contour_plot` draws.
    """
    fs = result.factors
    names = fs.names
    if x == y:
        raise ValueError("x and y must be two different factors")
    for axis in (x, y):
        if axis not in names:
            raise KeyError(axis)

    fixed = dict(fixed or {})
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(f"fixed refers to unknown factors: {sorted(unknown)}")

    cx = np.linspace(-1.0, 1.0, resolution)
    cy = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y = np.meshgrid(cx, cy)  # (resolution, resolution)

    coded: dict[str, np.ndarray] = {}
    for name in names:
        if name == x:
            coded[name] = grid_x
        elif name == y:
            coded[name] = grid_y
        elif name in fixed:
            coded_value = fs[name].code(np.asarray(fixed[name]))  # type: ignore[union-attr]
            coded[name] = np.full_like(grid_x, coded_value)
        else:
            coded[name] = np.zeros_like(grid_x)  # held at center

    z = _predict(result, coded, grid_x)
    nat_x = fs[x].decode(grid_x)  # type: ignore[union-attr]
    nat_y = fs[y].decode(grid_y)  # type: ignore[union-attr]
    return nat_x, nat_y, z


def contour_plot(
    result: FitResult,
    x: str,
    y: str,
    *,
    fixed: dict[str, float] | None = None,
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
    fixed: dict[str, float] | None = None,
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


def residuals_vs_fitted(result: FitResult, ax: Axes | None = None) -> Axes:
    """Scatter of residuals against fitted values, with a zero reference line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.scatter(result.fitted, result.residuals)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    return ax


def normal_qq(result: FitResult, ax: Axes | None = None) -> Axes:
    """Normal quantile-quantile plot of the residuals (``scipy.stats.probplot``)."""
    import matplotlib.pyplot as plt
    from scipy import stats

    if ax is None:
        _, ax = plt.subplots()
    stats.probplot(result.residuals, plot=ax)
    ax.set_title("Normal Q-Q")
    return ax


def half_normal_plot(result: FitResult, ax: Axes | None = None) -> Axes:
    """Absolute effects against half-normal quantiles (screening companion to pareto_plot)."""
    import matplotlib.pyplot as plt
    from scipy import stats

    if ax is None:
        _, ax = plt.subplots()

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
