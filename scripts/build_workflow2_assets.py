"""Run the docs/WORKFLOW2.md walkthrough, capture real outputs and save its figures.

This is the provenance for the outputs/figures embedded in docs/WORKFLOW2.md: it
reproduces the two-readout desirability example verbatim (same factors, same seeds),
prints every console block for transcription, and writes the figures to docs/img/ with
a ``wf2_`` prefix. Run with: uv run python scripts/build_workflow2_assets.py

The story is a deliberate sequel to WORKFLOW.md: there a single yield was maximized;
here yield (maximize) fights impurity (minimize), and Derringer-Suich desirability finds
the balance. Both responses are synthetic-but-realistic quadratic surfaces so the doc is
fully reproducible.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doe import (
    ContinuousFactor,
    ResponseGoal,
    central_composite,
    desirability,
    fit_ols,
)
from doe.plotting import contour_plot, surface_grid

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
# 1-2. Two competing readouts, factors, and the design
# --------------------------------------------------------------------------- #
banner("Sections 1-2: factors and central composite design")

factors = [
    ContinuousFactor("temperature", low=60, high=100, units="C"),
    ContinuousFactor("time", low=30, high=90, units="min"),
]

design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260708)
print(design.n_runs, design.n_center)
print(design.runs.head(8).round(2))


# --------------------------------------------------------------------------- #
# 3. Attach the two (synthetic) responses
# --------------------------------------------------------------------------- #
banner("Section 3: attach the two responses")

coded = design.coded()
t = coded["temperature"]
m = coded["time"]

rng = np.random.default_rng(2026)
# Yield: a dome that keeps climbing toward the hot/long corner (favours more of both).
yield_pct = (
    82.0
    + 6.0 * t
    + 4.5 * m
    - 4.0 * t**2
    - 3.0 * m**2
    + 1.5 * t * m
    + rng.normal(0, 0.7, design.n_runs)
)
# Impurity: rises monotonically with heat and time -- the cost of pushing the reaction.
impurity_pct = (
    8.0
    + 3.5 * t
    + 2.5 * m
    + 0.8 * t**2
    + 0.5 * m**2
    + rng.normal(0, 0.4, design.n_runs)
)

measured = design.with_responses(yield_pct=yield_pct, impurity_pct=impurity_pct)
print(measured.runs.head(8).round(2))


# --------------------------------------------------------------------------- #
# 4. Fit one model per readout
# --------------------------------------------------------------------------- #
banner("Section 4: a quadratic fit for each readout")

fit_yield = fit_ols(measured, "yield_pct", model="quadratic")
fit_imp = fit_ols(measured, "impurity_pct", model="quadratic")

for name, fit in (("yield_pct", fit_yield), ("impurity_pct", fit_imp)):
    print(f"\n{name}: R2={fit.r_squared:.3f}  adjR2={fit.adjusted_r2():.3f}  "
          f"predR2={fit.predicted_r2():.3f}")
    print(fit.summary().round(2))


# --------------------------------------------------------------------------- #
# 5. See the conflict: each readout pulls the other way
# --------------------------------------------------------------------------- #
banner("Section 5: the conflict between the single-readout optima")

opt_yield = fit_yield.optimum(maximize=True)
opt_imp = fit_imp.optimum(maximize=False)
print(f"yield-only optimum:    {opt_yield}")
print(f"impurity-only optimum: {opt_imp}")

# cross-read: what the *other* readout does at each single-readout optimum.
imp_at_yield_opt = fit_imp.predict(opt_yield.natural)
yield_at_imp_opt = fit_yield.predict(opt_imp.natural)
print(
    f"\nAt the yield-only optimum ({opt_yield.natural['temperature']:.0f} C, "
    f"{opt_yield.natural['time']:.0f} min): yield={opt_yield.response:.1f}%, "
    f"impurity={imp_at_yield_opt:.1f}%"
)
print(
    f"At the impurity-only optimum ({opt_imp.natural['temperature']:.0f} C, "
    f"{opt_imp.natural['time']:.0f} min): impurity={opt_imp.response:.1f}%, "
    f"yield={yield_at_imp_opt:.1f}%"
)

# Figure: the two fitted surfaces side by side, each with its own optimum marked. The
# yield peak sits toward the hot/long corner; the impurity minimum sits at the cool/short
# corner. Opposite corners = the trade-off, drawn.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))
contour_plot(fit_yield, "temperature", "time", ax=axL, resolution=200)
axL.scatter([opt_yield.natural["temperature"]], [opt_yield.natural["time"]],
            s=280, marker="*", color="gold", edgecolors="black", linewidths=1.2, zorder=6)
axL.set_title("Yield (%) -- maximise")
contour_plot(fit_imp, "temperature", "time", ax=axR, resolution=200)
axR.scatter([opt_imp.natural["temperature"]], [opt_imp.natural["time"]],
            s=200, marker="v", color="magenta", edgecolors="black", linewidths=1.2, zorder=6,
            clip_on=False)
axR.set_title("Impurity (%) -- minimise")
save(axL, "wf2_conflict.png")


# --------------------------------------------------------------------------- #
# 6. State the goals as desirability ramps
# --------------------------------------------------------------------------- #
banner("Section 6: desirability goals")

goals = [
    ResponseGoal(fit_yield, goal="max", low=78.0, high=88.0),   # yield: 0 below 78, 1 at 88
    ResponseGoal(fit_imp, goal="min", low=5.0, high=12.0),      # impurity: 1 below 5, 0 at 12
]
for g, label in zip(goals, ("yield_pct (max)", "impurity_pct (min)"), strict=True):
    print(f"{label}: low={g.low}, high={g.high}, weight={g.weight}")

# Figure: the two desirability ramps -- how each raw readout is mapped onto a common 0..1
# ruler. This is the modelling choice the whole balance rests on.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2))
yv = np.linspace(72, 90, 300)
axL.plot(yv, [goals[0].desirability(v) for v in yv], color="tab:green", lw=2.2)
axL.set_title("Yield desirability (maximise)")
axL.set_xlabel("yield (%)")
axL.set_ylabel("desirability d")
iv = np.linspace(2, 16, 300)
axR.plot(iv, [goals[1].desirability(v) for v in iv], color="tab:red", lw=2.2)
axR.set_title("Impurity desirability (minimise)")
axR.set_xlabel("impurity (%)")
axR.set_ylabel("desirability d")
for ax in (axL, axR):
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
save(axL, "wf2_ramps.png")


# --------------------------------------------------------------------------- #
# 7. Find the balanced operating point
# --------------------------------------------------------------------------- #
banner("Section 7: the balanced operating point")

des = desirability(goals)
print(des)
print(f"\nnatural     = {{'temperature': {des.natural['temperature']:.1f}, "
      f"'time': {des.natural['time']:.1f}}}")
print("responses   =", des.responses.round(1))
print("individual  =", des.individual.round(3))
print(f"overall D   = {des.overall:.3f}")

# Figure: the combined desirability surface D over the two factors, with all three optima
# marked. The gold star (balance) sits on the bright plateau; the yield-only X has slid
# off it into low-D territory because its impurity is unacceptable.
gx, gy, z_yield = surface_grid(fit_yield, "temperature", "time", resolution=201)
_, _, z_imp = surface_grid(fit_imp, "temperature", "time", resolution=201)
d_yield = np.vectorize(goals[0].desirability)(z_yield)
d_imp = np.vectorize(goals[1].desirability)(z_imp)
d_grid = np.sqrt(np.clip(d_yield, 0.0, None) * np.clip(d_imp, 0.0, None))

fig, ax = plt.subplots(figsize=(7.6, 5.6))
cf = ax.contourf(gx, gy, d_grid, levels=np.linspace(0.0, float(d_grid.max()), 13), cmap="viridis")
fig.colorbar(cf, ax=ax, label="overall desirability D")
ax.contour(gx, gy, d_grid, levels=8, colors="white", linewidths=0.4, alpha=0.5)
ax.scatter([des.natural["temperature"]], [des.natural["time"]], s=300, marker="*",
           color="gold", edgecolors="black", linewidths=1.2, zorder=6,
           label=f"balance (D={des.overall:.2f})")
ax.scatter([opt_yield.natural["temperature"]], [opt_yield.natural["time"]], s=150, marker="X",
           color="crimson", edgecolors="white", linewidths=1.2, zorder=6,
           label="yield-only optimum")
ax.scatter([opt_imp.natural["temperature"]], [opt_imp.natural["time"]], s=140, marker="v",
           color="magenta", edgecolors="black", linewidths=1.0, zorder=6,
           label="impurity-only optimum", clip_on=False)
ax.set_xlabel("temperature (C)")
ax.set_ylabel("time (min)")
ax.set_title("Desirability balances yield against impurity")
ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a1a", edgecolor="white",
          framealpha=0.85, labelcolor="white")
save(ax, "wf2_desirability.png")


# --------------------------------------------------------------------------- #
# 8. Tune the trade-off + confirmation
# --------------------------------------------------------------------------- #
banner("Section 8: tune the trade-off and confirm")

# A stricter purity requirement: refuse anything above 9% impurity, and demand getting
# close to ideal (weight 2). Watch the balance move toward cooler/shorter conditions.
strict = [
    ResponseGoal(fit_yield, goal="max", low=78.0, high=88.0),
    ResponseGoal(fit_imp, goal="min", low=5.0, high=9.0, weight=2.0),
]
des_strict = desirability(strict)
print("stricter purity goal (impurity high=9, weight=2):")
print(f"  natural   = {{'temperature': {des_strict.natural['temperature']:.1f}, "
      f"'time': {des_strict.natural['time']:.1f}}}")
print("  responses =", des_strict.responses.round(1))
print(f"  overall D = {des_strict.overall:.3f}")

# Figure: re-strike the same D surface under the stricter goals and show the plateau
# shrink and slide toward the cool/short corner. The original balance (gold star) now
# sits off the shrunken plateau; the new stricter balance (orange star) is on it.
d_yield_strict = np.vectorize(strict[0].desirability)(z_yield)
d_imp_strict = np.vectorize(strict[1].desirability)(z_imp)
d_grid_strict = np.sqrt(np.clip(d_yield_strict, 0.0, None) * np.clip(d_imp_strict, 0.0, None))

fig, ax = plt.subplots(figsize=(7.6, 5.6))
cf = ax.contourf(gx, gy, d_grid_strict, levels=np.linspace(0.0, float(d_grid_strict.max()), 13),
                  cmap="viridis")
fig.colorbar(cf, ax=ax, label="overall desirability D (strict goals)")
ax.contour(gx, gy, d_grid_strict, levels=8, colors="white", linewidths=0.4, alpha=0.5)
ax.scatter([des.natural["temperature"]], [des.natural["time"]], s=220, marker="*",
           color="gold", edgecolors="black", linewidths=1.0, zorder=6,
           label=f"original balance (D={des.overall:.2f})")
ax.scatter([des_strict.natural["temperature"]], [des_strict.natural["time"]], s=300,
           marker="*", color="darkorange", edgecolors="black", linewidths=1.2, zorder=7,
           label=f"stricter balance (D={des_strict.overall:.2f})")
ax.set_xlabel("temperature (C)")
ax.set_ylabel("time (min)")
ax.set_title("Tightening the impurity goal moves the balance")
ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a1a", edgecolor="white",
          framealpha=0.85, labelcolor="white")
save(ax, "wf2_tuned.png")

# Confirmation run at the (balanced) operating point.
confirmation = des.to_frame().round(1)
print("\nconfirmation run:")
print(confirmation)
print(f"predicted yield    = {fit_yield.predict(des.natural):.1f}%")
print(f"predicted impurity = {fit_imp.predict(des.natural):.1f}%")
print("\nyield 95% prediction interval:")
print(fit_yield.predict(des.natural, interval="prediction").round(1))
print("impurity 95% prediction interval:")
print(fit_imp.predict(des.natural, interval="prediction").round(1))

print("\nDONE")
