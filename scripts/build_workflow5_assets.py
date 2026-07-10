"""Run the docs/WORKFLOW5.md walkthrough, capture real outputs and save its figures.

This is the provenance for the outputs/figures embedded in docs/WORKFLOW5.md: it
reproduces the mixture (formulation) example verbatim (same components, same seeds),
prints every console block for transcription, and writes the figures to docs/img/ with
a ``wf5_`` prefix. Run with: uv run python scripts/build_workflow5_assets.py

The story is deliberately different from WORKFLOW.md/2/3: those all live on a *box* region
with continuous factors. Here the factors are *proportions of a whole* that must sum to 1,
so the design region is a constrained simplex, the model is a no-intercept Scheffe blending
polynomial, and the map is a ternary contour. The theme extends WORKFLOW/Vignette 21's
transfection example one layer down: not "how much reagent" but "which blend of lipids".
The response is a synthetic-but-realistic Scheffe surface so the doc is fully reproducible.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doe import MixtureFactor, fit_ols
from doe.generators.mixture import extreme_vertices, mixture_candidates, simplex_lattice
from doe.plotting import ternary_contour

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

HEIGHT = np.sqrt(3.0) / 2.0


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


def bary_to_xy(props: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ion, dope, chol) proportions -> Cartesian plot coords, matching ternary_contour."""
    x = props[:, 1] + 0.5 * props[:, 2]
    y = HEIGHT * props[:, 2]
    return x, y


# --------------------------------------------------------------------------- #
# 1. The components: proportions of a whole, with formulation bounds
# --------------------------------------------------------------------------- #
banner("Section 1: the three lipid components (a mixture, not a box)")

components = [
    MixtureFactor("ion", low=0.20, high=0.80),   # ionizable (cationic) lipid
    MixtureFactor("dope", low=0.05, high=0.50),  # helper phospholipid (DOPE)
    MixtureFactor("chol", low=0.10, high=0.60),  # cholesterol
]
lows = np.array([c.low for c in components])
highs = np.array([c.high for c in components])
print("component bounds (proportion of total lipid):")
for c in components:
    print(f"  {c.name:5s}  [{c.low:.2f}, {c.high:.2f}]")
print(f"sum of lows  = {lows.sum():.2f}  (must be <= 1)")
print(f"sum of highs = {highs.sum():.2f}  (must be >= 1)")

# Why not a lattice? A full simplex design would demand infeasible pure blends.
try:
    simplex_lattice(components, degree=2)
except ValueError as exc:
    print(f"\nsimplex_lattice(..., degree=2) -> ValueError:\n  {exc}")


# --------------------------------------------------------------------------- #
# 2. The design: extreme vertices of the bounded simplex, run in duplicate
# --------------------------------------------------------------------------- #
banner("Section 2: extreme-vertices design on the constrained simplex")

base = extreme_vertices(components)
print(f"base design: {base.n_runs} runs ({base.meta['n_vertices']} vertices + centroid)")
print(base.runs.round(3).to_string())
print("point types:", base.point_types)

# Duplicate every blend so pure error (and a lack-of-fit test) is available, then randomize.
design = base.replicate(2, each=True).randomize(seed=20260710)
print(f"\nreplicated + randomized: {design.n_runs} runs")


# --------------------------------------------------------------------------- #
# 3. Attach the response: transfection efficiency of each blend
# --------------------------------------------------------------------------- #
banner("Section 3: attach the measured response")

props = design.runs[[c.name for c in components]].to_numpy(dtype=float)
ion, dope, chol = props[:, 0], props[:, 1], props[:, 2]

# "Truth": a Scheffe quadratic blending surface. Pure components transfect poorly; the payoff
# is synergy -- especially the ionizable-lipid x cholesterol cross-term (stability + delivery).
rng = np.random.default_rng(20260710)
efficiency = (
    20.0 * ion
    + 15.0 * dope
    + 20.0 * chol
    + 30.0 * ion * dope
    + 170.0 * ion * chol  # the dominant ionizable-lipid x cholesterol synergy
    + 25.0 * dope * chol
    + rng.normal(0.0, 1.2, design.n_runs)
)
measured = design.with_responses(transfection=efficiency)
print(measured.runs.round(3).head(8).to_string())


# --------------------------------------------------------------------------- #
# 4. Fit a Scheffe blending model -- linear first, then quadratic
# --------------------------------------------------------------------------- #
banner("Section 4: Scheffe blending fits (linear vs quadratic)")

fit_lin = fit_ols(measured, "transfection", model="scheffe-linear")
fit_quad = fit_ols(measured, "transfection", model="scheffe-quadratic")

print(f"scheffe-linear    R2 = {fit_lin.r_squared:.3f}   adjR2 = {fit_lin.adjusted_r2():.3f}")
print(f"scheffe-quadratic R2 = {fit_quad.r_squared:.3f}   adjR2 = {fit_quad.adjusted_r2():.3f}")
print("\nquadratic blending coefficients:")
print(fit_quad.summary().round(2))

print("\nANOVA (mixture convention -- one 'Linear blending' row, then each cross product):")
print(fit_quad.anova().round(3))

lof = fit_quad.lack_of_fit()
print(f"\nlack-of-fit: F = {lof.f_stat:.3f}, p = {lof.p_value:.3f} "
      f"(pure-error df = {lof.df_pe}, LOF df = {lof.df_lof})")


# --------------------------------------------------------------------------- #
# 5. The two maps: the feasible region, and the fitted blending surface
# --------------------------------------------------------------------------- #
banner("Section 5: the ternary maps")

# Figure A: the constrained region -- the feasible polygon carved out of the full simplex by
# the component bounds, with the design's vertices and centroid marked.
vertices = base.runs[[c.name for c in components]].to_numpy(dtype=float)[:-1]  # drop centroid
centroid = base.runs[[c.name for c in components]].to_numpy(dtype=float)[-1]
vx, vy = bary_to_xy(vertices)
cx, cy = bary_to_xy(centroid[None, :])
order = np.argsort(np.arctan2(vy - cy, vx - cx))  # order polygon vertices by angle
poly_x = np.append(vx[order], vx[order][0])
poly_y = np.append(vy[order], vy[order][0])

fig, ax = plt.subplots(figsize=(6.4, 5.8))
tri_x = [0.0, 1.0, 0.5, 0.0]
tri_y = [0.0, 0.0, HEIGHT, 0.0]
ax.plot(tri_x, tri_y, color="0.6", lw=1.2)
ax.fill(poly_x, poly_y, color="tab:blue", alpha=0.15, zorder=1)
ax.plot(poly_x, poly_y, color="tab:blue", lw=1.6, zorder=2)
ax.scatter(vx, vy, s=70, color="crimson", edgecolors="white", linewidths=1.2, zorder=4,
           label="extreme vertices")
ax.scatter(cx, cy, s=150, marker="*", color="gold", edgecolors="black", linewidths=1.0,
           zorder=5, label="centroid")
for name, vxi, vyi, dx, dy, ha in [
    ("ion", 0.0, 0.0, -8, -12, "right"),
    ("dope", 1.0, 0.0, 8, -12, "left"),
    ("chol", 0.5, HEIGHT, 0, 8, "center"),
]:
    ax.annotate(name, (vxi, vyi), textcoords="offset points", xytext=(dx, dy), ha=ha)
ax.set_aspect("equal")
ax.set_axis_off()
ax.margins(0.08)
ax.set_title("The bounds carve a feasible polygon out of the simplex", pad=16)
ax.legend(loc="upper right", fontsize=8)
save(ax, "wf5_region.png")

# Figure B: the fitted blending surface over the whole simplex, with the design blends on it.
ax = ternary_contour(fit_quad, measured, resolution=160)
ax.set_title("Fitted transfection surface (Scheffe quadratic)", pad=22)
save(ax, "wf5_ternary.png")


# --------------------------------------------------------------------------- #
# 6. Find the best feasible blend and confirm it
# --------------------------------------------------------------------------- #
banner("Section 6: the best feasible blend")

candidates = mixture_candidates(components, resolution=50)
cand_df = pd.DataFrame({c.name: candidates[:, j] for j, c in enumerate(components)})
pred = np.asarray(fit_quad.predict(cand_df), dtype=float)
best_i = int(np.argmax(pred))
best = candidates[best_i]
print(f"searched {len(candidates)} feasible candidate blends")
print(f"best blend: ion={best[0]:.2f}, dope={best[1]:.2f}, chol={best[2]:.2f}")
print(f"predicted transfection = {pred[best_i]:.1f}%")

best_map = {c.name: float(best[j]) for j, c in enumerate(components)}
print("\n95% prediction interval at the best blend:")
print(fit_quad.predict(best_map, interval="prediction").round(1))

# For contrast: the best single-vertex (corner) blend the design actually ran.
vpred = np.asarray(
    fit_quad.predict(base.runs.iloc[:-1][[c.name for c in components]]), dtype=float
)
bv = int(np.argmax(vpred))
best_vertex = base.runs.iloc[:-1][[c.name for c in components]].to_numpy(float)[bv]
print(f"\nbest *vertex* blend: ion={best_vertex[0]:.2f}, dope={best_vertex[1]:.2f}, "
      f"chol={best_vertex[2]:.2f} -> {vpred[bv]:.1f}%")
print("The recommendation is a balanced ion:chol edge with the helper at its floor -- the")
print("ion x chol synergy term drives it, and a linear blending model would never have found it.")

# Figure C: the fitted surface again, with the optimal blend starred.
ax = ternary_contour(fit_quad, resolution=160)
bx, by = bary_to_xy(best[None, :])
ax.scatter(bx, by, s=320, marker="*", color="gold", edgecolors="black", linewidths=1.3,
           zorder=6, label="best feasible blend")
ax.set_title("Optimal transfection blend", pad=22)
ax.legend(loc="upper right", fontsize=8)
save(ax, "wf5_optimum.png")

print("\nDONE")
