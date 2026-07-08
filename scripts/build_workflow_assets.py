"""Run the docs/WORKFLOW.md walkthrough, capture real outputs and save its figures.

This is the provenance for the outputs/figures embedded in docs/WORKFLOW.md: it
reproduces the end-to-end reaction-optimization example verbatim (same factors, same
seeds), prints every console block for transcription, and writes the figures to
docs/img/ with a ``wf_`` prefix. Run with: uv run python scripts/build_workflow_assets.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doe import ContinuousFactor, central_composite, fit_ols, vif
from doe.plotting import (
    contour_plot,
    pareto_plot,
    predicted_vs_actual,
    residuals_vs_fitted,
    surface_grid,
)

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)


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
# 1-2. Define the space, generate + randomize the design
# --------------------------------------------------------------------------- #
banner("Sections 1-2: factors and central composite design")

factors = [
    ContinuousFactor("temperature", low=45, high=75, units="C"),
    ContinuousFactor("time", low=20, high=60, units="min"),
    ContinuousFactor("catalyst", low=0.5, high=2.5, units="g/L"),
]

design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260707)
print(design.n_runs, design.n_center)
print(design.runs.head(8).round(2))

# Figure: what a faced central composite design actually looks like in the coded cube.
# Factorial corners span the box, axial points sit on the face centers (curvature
# information), and the replicated center anchors pure error / lack-of-fit.
coded = design.coded()
pts = coded.to_numpy(dtype=float)
kinds = np.array(design.point_types)
style = {
    "factorial": ("tab:blue", "o", 55, "factorial corners (8)"),
    "axial": ("tab:orange", "^", 80, "axial / face points (6)"),
    "center": ("tab:red", "D", 70, "center replicates (5)"),
}
fig = plt.figure(figsize=(7.0, 6.2))
ax3d = fig.add_subplot(projection="3d")
# faint wireframe of the coded cube edges
r = [-1.0, 1.0]
for s in r:
    for t in r:
        ax3d.plot([s, s], [t, t], r, color="0.8", lw=0.7, zorder=1)
        ax3d.plot([s, s], r, [t, t], color="0.8", lw=0.7, zorder=1)
        ax3d.plot(r, [s, s], [t, t], color="0.8", lw=0.7, zorder=1)
for kind, (color, marker, size, label) in style.items():
    m = kinds == kind
    ax3d.scatter(
        pts[m, 0], pts[m, 1], pts[m, 2],
        color=color, marker=marker, s=size, depthshade=False,
        edgecolor="white", linewidths=0.6, label=label, zorder=3,
    )
ax3d.set_xlabel("temperature (coded)")
ax3d.set_ylabel("time (coded)")
ax3d.set_zlabel("catalyst (coded)")
ax3d.set_title("Faced central composite design: 19 runs, 3 factors")
ax3d.legend(loc="upper left", fontsize=8)
ax3d.view_init(elev=18, azim=-60)
save(ax3d, "wf_design.png")


# --------------------------------------------------------------------------- #
# 3. Attach the (synthetic) response
# --------------------------------------------------------------------------- #
banner("Section 3: attach the response")

rng = np.random.default_rng(42)
yield_pct = (
    78
    + 7.5 * coded["temperature"]
    + 5.0 * coded["time"]
    + 3.0 * coded["catalyst"]
    - 8.0 * coded["temperature"] ** 2
    - 5.5 * coded["time"] ** 2
    - 4.0 * coded["catalyst"] ** 2
    + 2.5 * coded["temperature"] * coded["time"]
    - 1.5 * coded["time"] * coded["catalyst"]
    + rng.normal(0, 0.8, design.n_runs)
)
measured = design.with_response("yield_pct", yield_pct)


# --------------------------------------------------------------------------- #
# 4. Fit the quadratic model
# --------------------------------------------------------------------------- #
banner("Section 4: quadratic fit")

fit = fit_ols(measured, "yield_pct", model="quadratic")
print(f"R2={fit.r_squared:.3f}")
print(f"adjusted R2={fit.adjusted_r2():.3f}")
print(f"predicted R2={fit.predicted_r2():.3f}")

summary = pd.DataFrame(fit.summary(), index=["coefficient", "effect"]).T
print(summary.round(2))

# Figure: a Pareto plot ranks the fitted terms by magnitude, so the dominant
# main effects and curvature stand out from the near-zero interactions.
ax = pareto_plot(fit)
ax.set_title("Term magnitudes (Pareto of standardized effects)")
save(ax, "wf_pareto.png")


# --------------------------------------------------------------------------- #
# 5. Check whether the model is usable
# --------------------------------------------------------------------------- #
banner("Section 5: model checks")

print(fit.anova().round(3))
lof = fit.lack_of_fit()
print(f"lack-of-fit p={lof.p_value:.3f}")
print(vif(fit.model_matrix, term_names=fit.term_names).round(2))

# Figure: the two headline goodness checks side by side. Predicted-vs-actual should
# hug the 45-degree line; residuals-vs-fitted should be a structureless band.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8))
predicted_vs_actual(fit, ax=axL)
residuals_vs_fitted(fit, ax=axR)
save(axL, "wf_diagnostics.png")


# --------------------------------------------------------------------------- #
# 6-7. Choose the operating point, plan the confirmation run
# --------------------------------------------------------------------------- #
banner("Sections 6-7: operating point")

stationary = fit.stationary_point()
optimum = fit.optimum(maximize=True)
print(stationary)
print(optimum)

confirmation = pd.DataFrame([optimum.natural]).round(2)
print(confirmation)
print(fit.predict(optimum.natural).round(2))

# Figure: the fitted surface over temperature x time at the optimal catalyst loading,
# with the recommended operating point (and its confirmation setting) marked. This is
# the payoff: where on the map the next run should go.
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
save(ax, "wf_operating_point.png")

# sanity: confirm the marked point is the grid argmax of the same slice
gx, gy, gz = surface_grid(fit, "temperature", "time", fixed={"catalyst": cat_opt}, resolution=200)
jmax, imax = np.unravel_index(int(np.argmax(gz)), gz.shape)
print(
    f"\ngrid argmax on the catalyst={cat_opt:.2f} slice: "
    f"temperature={gx[jmax, imax]:.1f}, time={gy[jmax, imax]:.1f}, yield={gz.max():.2f}"
)

print("\nDONE")
