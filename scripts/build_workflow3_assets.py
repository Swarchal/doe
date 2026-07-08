"""Run the docs/WORKFLOW3.md walkthrough, capture real outputs and save its figures.

This is the provenance for the outputs/figures embedded in docs/WORKFLOW3.md: it
reproduces the screen-then-augment campaign verbatim (same factors, same seeds), prints
every console block for transcription, and writes the figures to docs/img/ with a
``wf3_`` prefix. Run with: uv run python scripts/build_workflow3_assets.py

The story is a deliberate prequel to WORKFLOW.md: that walkthrough started from the
three factors that matter; this one starts from the six candidate factors you actually
face, screens them in 16 runs, and recycles those runs into the curvature design --
arriving at the same operating point for a fraction of the one-shot cost. The response
is the same synthetic-but-realistic quadratic surface as WORKFLOW.md (three real
factors), with three inert factors that contribute only a whisper.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doe import (
    ContinuousFactor,
    augment,
    central_composite,
    efficiency,
    fit_ols,
    fractional_factorial,
)
from doe.plotting import (
    contour_plot,
    half_normal_plot,
    main_effects_plot,
    pareto_plot,
    predicted_vs_actual,
    residuals_vs_fitted,
)

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 110)


def save(ax: object, name: str) -> None:
    fig = ax.figure  # type: ignore[attr-defined]
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1. Six suspects, one run budget
# --------------------------------------------------------------------------- #
banner("Section 1: six candidate factors and what a one-shot design would cost")

factors = [
    ContinuousFactor("temperature", low=45, high=75, units="C"),
    ContinuousFactor("time", low=20, high=60, units="min"),
    ContinuousFactor("catalyst", low=0.5, high=2.5, units="g/L"),
    ContinuousFactor("pH", low=6.0, high=8.0),
    ContinuousFactor("stir_rate", low=300, high=900, units="rpm"),
    ContinuousFactor("solvent", low=10, high=30, units="%"),
]

one_shot = central_composite(factors, alpha="faced", center=5)
print(one_shot.n_runs)


# --------------------------------------------------------------------------- #
# 2. A 16-run screen that varies everything at once
# --------------------------------------------------------------------------- #
banner("Section 2: the 2^(6-2) screening design")

screen = fractional_factorial(
    factors, generators=["E=ABC", "F=BCD"]
).randomize(seed=20260709)
print(screen.n_runs)
print(screen.runs.head(8).round(2))

# Figure: the coded screen matrix as a red/blue tile map (standard order). Every column
# is half high / half low, and any two columns pair each level of one with both levels
# of the other equally often -- the balance that lets 16 runs judge 6 factors at once.
coded_screen = screen.coded()
std = screen.runs["std_order"].to_numpy()
tiles = coded_screen.to_numpy(dtype=float)[np.argsort(std)]
fig, ax = plt.subplots(figsize=(6.2, 6.4))
ax.imshow(tiles, cmap="coolwarm", vmin=-1.35, vmax=1.35, aspect="auto")
for i in range(tiles.shape[0]):
    for j in range(tiles.shape[1]):
        ax.text(
            j, i, "+" if tiles[i, j] > 0 else "−",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )
ax.set_xticks(range(len(coded_screen.columns)))
ax.set_xticklabels(coded_screen.columns, rotation=30, ha="right")
ax.set_yticks(range(tiles.shape[0]))
ax.set_yticklabels([str(i + 1) for i in range(tiles.shape[0])], fontsize=8)
ax.set_ylabel("run (standard order)")
ax.set_title("The screening plan: 16 runs, 6 factors,\nevery factor high in half the runs")
save(ax, "wf3_screen_layout.png")


# --------------------------------------------------------------------------- #
# 3. Run the screen and read the effects
# --------------------------------------------------------------------------- #
banner("Section 3: attach yields and fit the screening model")


def bench_yield(coded_runs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Stand-in for the lab bench: the 'true' yield surface plus run-to-run noise.

    Only three of the six knobs actually do anything; pH, stir_rate, and solvent
    contribute a whisper (folded into the noise once they are dropped and held fixed).
    Replace this with your own measurements.
    """
    t, m, c = coded_runs["temperature"], coded_runs["time"], coded_runs["catalyst"]
    y = (
        78
        + 7.5 * t + 5.0 * m + 3.0 * c
        - 8.0 * t**2 - 5.5 * m**2 - 4.0 * c**2
        + 2.5 * t * m - 1.5 * m * c
    )
    for name, beta in (("pH", 0.4), ("stir_rate", -0.3), ("solvent", 0.35)):
        if name in coded_runs:
            y = y + beta * coded_runs[name]
    return np.asarray(y + rng.normal(0, 0.8, len(coded_runs)))


rng = np.random.default_rng(20260709)
measured = screen.with_response("yield_pct", bench_yield(screen.coded(), rng))
print(measured.runs.head(8).round(2))

screen_fit = fit_ols(measured, "yield_pct", interactions=False)
print(f"\nR2={screen_fit.r_squared:.3f}")
print(screen_fit.summary().round(2))

# Figure: the half-normal plot -- the screening verdict in one picture. Inactive factors
# scatter along the noise line through the origin; the real ones break above it.
ax = half_normal_plot(screen_fit)
ax.set_xlim(right=2.05)  # room for the rightmost label
save(ax, "wf3_half_normal.png")

# Figure: the same verdict two more ways -- ranked bar sizes (Pareto) and signed
# slopes (main effects). Three tall bars / steep slopes, three stubs / flat lines.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
pareto_plot(screen_fit, ax=axL)
main_effects_plot(screen_fit, ax=axR)
axR.tick_params(axis="x", rotation=30)
save(axL, "wf3_screen_effects.png")


# --------------------------------------------------------------------------- #
# 4-5. Keep the vital few, recycle the screen into a curvature design
# --------------------------------------------------------------------------- #
banner("Sections 4-5: project onto the survivors and augment")

keep = ["temperature", "time", "catalyst"]
projected = measured.project(keep)
survivors = projected.factors

print(projected.coded().value_counts().sort_index())

eff_before = efficiency(projected, order=2)
print(f"\nD-efficiency for a quadratic model, screen runs only: {eff_before.d:.3f}")

augmented = augment(projected, n_runs=8, model="quadratic", seed=20260710)
eff_after = efficiency(augmented, order=2)
print(f"D-efficiency for a quadratic model, after augmenting:  {eff_after.d:.3f}")

# sanity for the prose comparison: the first walkthrough's faced CCD on the same three
# factors scores *lower* on the same quadratic model.
ccd_eff = efficiency(central_composite(list(survivors), alpha="faced", center=5), order=2)
print(f"(for comparison, a fresh 19-run faced CCD scores D={ccd_eff.d:.3f})")

new_runs = augmented.runs.assign(point_type=augmented.point_types).tail(8)
print(f"\n{augmented.n_runs} runs total; the 8 new runs to go measure:")
print(new_runs.round(2))

# Figure: the recycled campaign in the coded cube. The screen contributed the corners
# (each measured twice); augment added the mid-level points a curved model needs.
pts = augmented.coded().to_numpy(dtype=float)
kinds = np.array(augmented.point_types)
fig = plt.figure(figsize=(7.0, 6.2))
ax3d = fig.add_subplot(projection="3d")
r = [-1.0, 1.0]
for s in r:
    for t in r:
        ax3d.plot([s, s], [t, t], r, color="0.8", lw=0.7, zorder=1)
        ax3d.plot([s, s], r, [t, t], color="0.8", lw=0.7, zorder=1)
        ax3d.plot(r, [s, s], [t, t], color="0.8", lw=0.7, zorder=1)
m = kinds == "existing"
ax3d.scatter(
    pts[m, 0], pts[m, 1], pts[m, 2],
    color="tab:blue", marker="o", s=60, depthshade=False,
    edgecolor="white", linewidths=0.6, label="screening corners (8 settings x 2)", zorder=3,
)
m = kinds == "augment"
ax3d.scatter(
    pts[m, 0], pts[m, 1], pts[m, 2],
    color="tab:orange", marker="^", s=90, depthshade=False,
    edgecolor="black", linewidths=0.6, label="new augmented runs (8)", zorder=4,
)
ax3d.set_xlabel("temperature (coded)")
ax3d.set_ylabel("time (coded)")
ax3d.set_zlabel("catalyst (coded)")
ax3d.set_title(
    "Recycling the screen: 16 corner runs kept,\n8 new runs add the curvature information"
)
ax3d.legend(loc="upper left", fontsize=8)
ax3d.view_init(elev=18, azim=-60)
save(ax3d, "wf3_augmented.png")


# --------------------------------------------------------------------------- #
# 6. Measure only the new runs and fit the quadratic
# --------------------------------------------------------------------------- #
banner("Section 6: measure the 8 new runs, fit the quadratic on all 24")

yield_new = bench_yield(augmented.coded().iloc[projected.n_runs :], rng)
all_yields = np.concatenate([measured.runs["yield_pct"].to_numpy(), yield_new])
measured2 = augmented.with_response("yield_pct", all_yields)

fit = fit_ols(measured2, "yield_pct", model="quadratic")
print(f"R2={fit.r_squared:.3f}")
print(f"adjusted R2={fit.adjusted_r2():.3f}")
print(f"predicted R2={fit.predicted_r2():.3f}")
print(fit.summary().round(2))

# Figure: the two headline goodness checks side by side, as in WORKFLOW.md.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8))
predicted_vs_actual(fit, ax=axL)
residuals_vs_fitted(fit, ax=axR)
save(axL, "wf3_diagnostics.png")


# --------------------------------------------------------------------------- #
# 7. The operating point -- and the bill
# --------------------------------------------------------------------------- #
banner("Section 7: operating point and campaign cost")

optimum = fit.optimum(maximize=True)
print(optimum)
print(optimum.to_frame().round(2))
print(fit.predict(optimum.natural, interval="prediction").round(2))

# Figure: the fitted surface at the optimal catalyst loading, operating point starred.
cat_opt = optimum.natural["catalyst"]
temp_opt = optimum.natural["temperature"]
time_opt = optimum.natural["time"]
ax = contour_plot(fit, "temperature", "time", fixed={"catalyst": cat_opt}, resolution=200)
ax.scatter(
    [temp_opt], [time_opt],
    s=300, marker="*", color="gold", edgecolors="black", linewidths=1.2, zorder=6,
)
ax.annotate(
    f"operating point\n{temp_opt:.0f} C, {time_opt:.0f} min → {optimum.response:.1f}% yield",
    xy=(temp_opt, time_opt), xytext=(temp_opt - 17, time_opt - 12),
    fontsize=9, color="black",
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85),
)
ax.set_title(f"Fitted yield surface at catalyst = {cat_opt:.2f} g/L")
save(ax, "wf3_operating_point.png")

# Figure: the bill for three routes to the same operating point. Stacked bars: the
# screening phase (shared by the two sequential routes) plus each route's follow-up.
fresh_ccd = central_composite(list(survivors), alpha="faced", center=5).n_runs
routes = [
    ("One-shot surface\non all six factors", 0, one_shot.n_runs),
    ("Screen, then fresh\nsurface on the three", screen.n_runs, fresh_ccd),
    ("Screen, then augment\n(this walkthrough)", screen.n_runs, 8),
]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
labels = [r[0] for r in routes]
screens = np.array([r[1] for r in routes], dtype=float)
follows = np.array([r[2] for r in routes], dtype=float)
y = np.arange(len(routes))[::-1]
ax.barh(y, screens, color="tab:blue", label="screening runs")
ax.barh(y, follows, left=screens, color="tab:orange", label="response-surface runs")
for yi, total in zip(y, screens + follows, strict=True):
    ax.text(total + 1, yi, f"{int(total)} runs", va="center", fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("total runs")
ax.set_xlim(0, 92)
ax.set_title("Three routes to the same operating point")
ax.legend(loc="lower right", fontsize=8)
save(ax, "wf3_budget.png")

total_fresh = screen.n_runs + fresh_ccd
print(f"\none-shot on six factors:        {one_shot.n_runs} runs")
print(f"screen + fresh CCD on three:    {screen.n_runs} + {fresh_ccd} = {total_fresh} runs")
print(f"screen + augment (this route):  {screen.n_runs} + 8 = {augmented.n_runs} runs")

print("\nDONE")
