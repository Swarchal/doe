"""Run every VIGNETTES.md example, capture real outputs and save the figures.

This is the provenance for the outputs/visualisations embedded in docs/VIGNETTES.md.
Run with: uv run python scripts/build_vignette_assets.py
Figures are written to docs/img/ and console outputs are printed for transcription.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doe import (
    ContinuousFactor,
    box_behnken,
    central_composite,
    fit_ols,
    fractional_factorial,
    full_factorial,
)
from doe.analysis import (
    adjusted_r2,
    anova_table,
    lack_of_fit,
    predicted_r2,
    press,
)
from doe.plotting import (
    contour_plot,
    half_normal_plot,
    main_effects_plot,
    normal_qq,
    pareto_plot,
    residuals_vs_fitted,
    surface_grid,
)

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)


def save(ax, name: str) -> None:
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Vignettes 1-3: 2x2 factorial, GFP readout
# --------------------------------------------------------------------------- #
banner("Vignettes 1-3: 2x2 factorial")

dna = ContinuousFactor("dna_ng", 100, 500, units="ng/well")
lipid = ContinuousFactor("lipid_uL", 0.5, 2.5, units="uL/well")
design = full_factorial([dna, lipid], levels=2)
print("design.runs:")
print(design.runs)

gfp = np.array(
    [22, 20, 24, 31, 29, 33, 40, 38, 42, 58, 60, 62],
    dtype=float,
)
rep = design.replicate(3, each=True)
result = fit_ols(rep, gfp, model="linear")
print("\nresult.summary():")
for k, (c, e) in result.summary().items():
    print(f"  {k!r}: (coef={c:.4g}, effect={e:.4g})")
print(f"\nR^2 = {result.r_squared:.4f}")

ax = main_effects_plot(result)
save(ax, "v2_main_effects.png")
ax = pareto_plot(result)
save(ax, "v3_pareto.png")


# --------------------------------------------------------------------------- #
# Vignette 4: fractional factorial screen, half-normal plot
# --------------------------------------------------------------------------- #
banner("Vignette 4: fractional factorial screen")

A = ContinuousFactor("seeding_cells", 5_000, 20_000, units="cells/well")
B = ContinuousFactor("serum_pct", 2, 10, units="%")
C = ContinuousFactor("dmso_pct", 0.1, 1.0, units="%")
D = ContinuousFactor("compound_uM", 0.1, 10, units="uM")
screen = fractional_factorial([A, B, C, D], generators=["D=ABC"])
print(f"screen.n_runs = {screen.n_runs}")
print("screen.coded():")
print(screen.coded())

# A screen where compound (D) and serum (B) dominate; seeding and DMSO are inert.
coded = screen.coded().to_numpy(dtype=float)
rng = np.random.default_rng(7)
true = 50 + 12 * coded[:, 3] + 7 * coded[:, 1]  # compound + serum
y_screen = np.round(true + rng.normal(0, 0.6, size=true.shape), 1)
print("\ny_screen:", list(y_screen))
res_screen = fit_ols(screen, y_screen, model="linear")
print("\nscreen effects (|effect|, sorted):")
order = np.argsort(np.abs(res_screen.effects))[::-1]
for i in order:
    if res_screen.term_names[i] == "Intercept":
        continue
    print(f"  {res_screen.term_names[i]:>28s}: {res_screen.effects[i]:+.3f}")

ax = half_normal_plot(res_screen)
save(ax, "v4_half_normal.png")


# --------------------------------------------------------------------------- #
# Vignettes 5-7: CCD, lack-of-fit, contour, diagnostics
# --------------------------------------------------------------------------- #
banner("Vignettes 5-7: central composite design")

ccd = central_composite([dna, lipid], alpha="faced", center=4)
print(f"ccd.n_runs = {ccd.n_runs}, ccd.n_center = {ccd.n_center}")
print("ccd.coded():")
print(ccd.coded())

# Ground-truth dome with an interior optimum, plus realistic well noise.
cc = ccd.coded().to_numpy(dtype=float)
xa, xb = cc[:, 0], cc[:, 1]
true_surface = 60 + 11 * xa + 7 * xb + 3 * xa * xb - 9 * xa**2 - 6 * xb**2
rng = np.random.default_rng(42)
y_ccd = true_surface + rng.normal(0, 1.0, size=true_surface.shape)
y_ccd = np.round(y_ccd, 1)
print("\ny_ccd (% GFP+ per run):")
print(y_ccd)

res_ccd = fit_ols(ccd, y_ccd, model="quadratic")
print("\nquadratic fit summary():")
for k, (c, e) in res_ccd.summary().items():
    print(f"  {k!r}: (coef={c:.4g}, effect={e:.4g})")
print(f"R^2 = {res_ccd.r_squared:.4f}")

lof = lack_of_fit(res_ccd, ccd, y_ccd)
print(
    f"\nlack_of_fit: F={lof.f_stat:.3f}, p_value={lof.p_value:.4f} "
    f"(df_lof={lof.df_lof}, df_pe={lof.df_pe})"
)

# contour + surface grid optimum
ax = contour_plot(res_ccd, "dna_ng", "lipid_uL")
save(ax, "v6_contour.png")

X, Y, Z = surface_grid(res_ccd, "dna_ng", "lipid_uL", resolution=101)
i, j = np.unravel_index(np.argmax(Z), Z.shape)
print(
    f"\npredicted optimum: {X[i, j]:.0f} ng DNA, {Y[i, j]:.2f} uL lipid "
    f"-> {Z[i, j]:.1f}% GFP+"
)

ax = residuals_vs_fitted(res_ccd)
save(ax, "v7_residuals_vs_fitted.png")
ax = normal_qq(res_ccd)
save(ax, "v7_normal_qq.png")


# --------------------------------------------------------------------------- #
# Vignette 8: ANOVA and significance (reuses the V6 CCD quadratic fit)
# --------------------------------------------------------------------------- #
banner("Vignette 8: ANOVA and significance")

tbl = anova_table(res_ccd, ccd, y_ccd)
print("anova_table(res_ccd, ccd, y_ccd):")
print(tbl.to_string(float_format=lambda v: f"{v:.4g}"))

print("\nper-term (effect, t, p) and 95% CI on the coefficient:")
ci = res_ccd.conf_int(0.95)
for name, eff, t, p, (lo, hi) in zip(
    res_ccd.term_names, res_ccd.effects, res_ccd.t_values, res_ccd.p_values, ci, strict=True
):
    print(f"  {name:>16s}: effect={eff:+7.3f}  t={t:+7.2f}  p={p:.4f}  CI=[{lo:+.3f}, {hi:+.3f}]")


# --------------------------------------------------------------------------- #
# Vignette 9: how much model is too much -- adjusted & predicted R^2
# --------------------------------------------------------------------------- #
banner("Vignette 9: adjusted & predicted R^2")

res_lin = fit_ols(ccd, y_ccd, model="linear")  # same data, flat (no curvature) model
print("linear (flat) model on the CCD data:")
print(f"  R^2           = {res_lin.r_squared:.4f}")
print(f"  adjusted R^2  = {adjusted_r2(res_lin):.4f}")
print(f"  predicted R^2 = {predicted_r2(res_lin):.4f}")
print(f"  PRESS         = {press(res_lin):.2f}")
print("\nquadratic model on the same data:")
print(f"  R^2           = {res_ccd.r_squared:.4f}")
print(f"  adjusted R^2  = {adjusted_r2(res_ccd):.4f}")
print(f"  predicted R^2 = {predicted_r2(res_ccd):.4f}")
print(f"  PRESS         = {press(res_ccd):.2f}")


# --------------------------------------------------------------------------- #
# Vignette 10: Box-Behnken -- a leaner 3-factor response surface
# --------------------------------------------------------------------------- #
banner("Vignette 10: Box-Behnken design")

serum = ContinuousFactor("serum_pct", 2, 10, units="%")
bb = box_behnken([dna, lipid, serum], center=3)
print(f"bb.n_runs = {bb.n_runs}, bb.n_center = {bb.n_center}")
print("bb.coded():")
print(bb.coded())

# Ground-truth dome in 3 factors with an interior optimum, plus well noise.
bc = bb.coded().to_numpy(dtype=float)
xa, xb, xc = bc[:, 0], bc[:, 1], bc[:, 2]
true_bb = (
    60 + 11 * xa + 7 * xb + 4 * xc + 3 * xa * xb - 9 * xa**2 - 6 * xb**2 - 5 * xc**2
)
rng = np.random.default_rng(11)
y_bb = np.round(true_bb + rng.normal(0, 1.0, size=true_bb.shape), 1)
print("\ny_bb (% GFP+ per run):")
print(y_bb)

res_bb = fit_ols(bb, y_bb, model="quadratic")
print("\nquadratic fit summary():")
for k, (c, e) in res_bb.summary().items():
    print(f"  {k!r}: (coef={c:.4g}, effect={e:.4g})")
print(f"R^2 = {res_bb.r_squared:.4f}, predicted R^2 = {predicted_r2(res_bb):.4f}")

ax = contour_plot(res_bb, "dna_ng", "lipid_uL", fixed={"serum_pct": 10})
save(ax, "v10_box_behnken_contour.png")


# --------------------------------------------------------------------------- #
# Vignette 11: randomise run order
# --------------------------------------------------------------------------- #
banner("Vignette 11: randomise run order")

plate_order = ccd.randomize(seed=42)
print("plate_order.runs.head():")
print(plate_order.runs.head())

print("\nDONE")
