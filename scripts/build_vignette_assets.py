"""Run every VIGNETTES.md example, capture real outputs and save the figures.

This is the provenance for the outputs/visualisations embedded in docs/VIGNETTES.md.
Run with: uv run python scripts/build_vignette_assets.py
Figures are written to docs/img/ and console outputs are printed for transcription.
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doe import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    FactorSet,
    MixtureFactor,
    ResponseGoal,
    augment,
    box_behnken,
    candidate_grid,
    central_composite,
    condition_number,
    d_optimal,
    desirability,
    discrepancy,
    efficiency,
    extreme_vertices,
    fit_ols,
    fractional_factorial,
    full_factorial,
    halton,
    i_optimal,
    latin_hypercube,
    maximin_distance,
    mixture_candidates,
    optimum,
    plackett_burman,
    simplex_centroid,
    simplex_lattice,
    sobol,
    stationary_point,
    to_html,
    vif,
)
from doe.analysis import (
    adjusted_r2,
    anova_table,
    lack_of_fit,
    predicted_r2,
    press,
)
from doe.plotting import (
    alias_matrix,
    contour_plot,
    correlation_heatmap,
    half_normal_plot,
    interaction_lines,
    interaction_plot,
    leverage_plot,
    main_effects_plot,
    normal_qq,
    pareto_plot,
    predicted_vs_actual,
    residuals_vs_fitted,
    surface_grid,
    surface_plot,
    ternary_contour,
    ternary_grid,
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


def show_nat(d: dict[str, float]) -> str:
    """Format a {dna_ng, lipid_uL} natural-units dict the way the vignettes transcribe it."""
    return f"{{'dna_ng': {d['dna_ng']:.1f}, 'lipid_uL': {d['lipid_uL']:.3f}}}"


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

# interaction plot: fitted %GFP+ vs DNA, one line per lipid level. Non-parallel lines
# are the dna x lipid interaction made visible. Print the line endpoints transcribed
# in the vignette.
nat_x, lines = interaction_lines(result, "dna_ng", "lipid_uL")
print("\ninteraction_lines(result, 'dna_ng', 'lipid_uL'):")
for level, z in lines:
    print(f"  lipid={level:g} uL: DNA 100->500 gives {z[0]:.1f} -> {z[-1]:.1f}% GFP+")
ax = interaction_plot(result, "dna_ng", "lipid_uL")
save(ax, "v3_interaction.png")

# Vignette 1 figure: OFAT vs factorial on a strongly-interacting surface.
# A synthetic ground-truth surface with a diagonal ridge (large positive interaction):
# OFAT walks the axes and stalls partway up the ridge, while the factorial samples all
# four corners and reveals the (high DNA, high lipid) direction the ridge actually climbs.
def true_v1(a: Any, b: Any) -> Any:  # coded units; accepts floats or arrays
    return 50 + 10 * (a + b) + 12 * a * b - 6 * a**2 - 6 * b**2


ga = np.linspace(-1.0, 1.0, 200)
gb = np.linspace(-1.0, 1.0, 200)
GA, GB = np.meshgrid(ga, gb)
Zv1 = true_v1(GA, GB)
NA, NB = dna.decode(GA), lipid.decode(GB)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
for ax in (axL, axR):
    ax.contourf(NA, NB, Zv1, levels=12, cmap="viridis")
    ax.contour(NA, NB, Zv1, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    ax.set_xlabel("dna_ng")
    ax.set_ylabel("lipid_uL")

# OFAT: start low/low, titrate DNA (lipid held low), then titrate lipid at the chosen DNA.
a1 = -1.0 / 6.0  # argmax over a of true_v1(a, -1)
b2 = (10.0 + 12.0 * a1) / 12.0  # argmax over b of true_v1(a1, b)
pa = dna.decode(np.array([-1.0, a1, a1]))
pb = lipid.decode(np.array([-1.0, -1.0, b2]))
axL.plot(pa, pb, "-", color="crimson", lw=2, zorder=5)
for k in range(2):
    axL.annotate("", xy=(pa[k + 1], pb[k + 1]), xytext=(pa[k], pb[k]),
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
axL.scatter(pa, pb, s=55, color="crimson", zorder=6)
axL.scatter([pa[2]], [pb[2]], s=180, facecolors="none", edgecolors="crimson",
            linewidths=2.5, zorder=7)
axL.text(pa[2], pb[2] + 0.13, f"OFAT stalls\n~{true_v1(a1, b2):.0f}% GFP+",
         color="crimson", fontsize=9, ha="center", va="bottom")
axL.set_title("One factor at a time\n(titrate DNA, then lipid)", fontsize=10)

ca = np.array([-1.0, -1.0, 1.0, 1.0])
cb = np.array([-1.0, 1.0, -1.0, 1.0])
axR.scatter(dna.decode(ca), lipid.decode(cb), s=110, color="white",
            edgecolors="black", linewidths=1.8, zorder=5)
best = int(np.argmax(true_v1(ca, cb)))
axR.scatter([dna.decode(ca[best])], [lipid.decode(cb[best])], s=240, marker="*",
            color="gold", edgecolors="black", linewidths=1.2, zorder=6)
axR.text(dna.decode(ca[best]), lipid.decode(cb[best]) - 0.16,
         f"best corner ~{true_v1(1.0, 1.0):.0f}%", color="black", fontsize=9,
         ha="center", va="top")
axR.set_title("Full factorial\n(all four corners)", fontsize=10)
axL.set_xlim(dna.decode(np.array([-1.08]))[0], dna.decode(np.array([1.08]))[0])
axL.set_ylim(lipid.decode(np.array([-1.08]))[0], lipid.decode(np.array([1.08]))[0])
print(f"\nOFAT stalls at ~{true_v1(a1, b2):.0f}% GFP+; the (high, high) corner reaches "
      f"~{true_v1(1.0, 1.0):.0f}%")
save(axL, "v1_ofat_vs_factorial.png")


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
# Vignette 5: Plackett-Burman -- the leanest main-effect screen
# --------------------------------------------------------------------------- #
banner("Vignette 5: Plackett-Burman screening")

pb_factors = [
    ContinuousFactor("seeding_cells", 5_000, 20_000, units="cells/well"),
    ContinuousFactor("serum_pct", 2, 10, units="%"),
    ContinuousFactor("dmso_pct", 0.1, 1.0, units="%"),
    ContinuousFactor("compound_uM", 0.1, 10, units="uM"),
    ContinuousFactor("incubation_h", 24, 72, units="h"),
    ContinuousFactor("passage_num", 5, 25, units="passage"),
    ContinuousFactor("dna_ng", 100, 500, units="ng/well"),
    ContinuousFactor("lipid_uL", 0.5, 2.5, units="uL/well"),
    ContinuousFactor("antibiotic_pct", 0.0, 1.0, units="%"),
    ContinuousFactor("coating_ugml", 1, 50, units="ug/mL"),
    ContinuousFactor("media_age_d", 1, 14, units="d"),
]
pb = plackett_burman(pb_factors)
print(f"pb.n_runs = {pb.n_runs} for {len(pb_factors)} factors")
print("pb.coded():")
print(pb.coded())

codedpb = pb.coded().to_numpy(dtype=float)
n_pb = codedpb.shape[0]
gram = codedpb.T @ codedpb
print(f"\nevery column balanced (sums to zero): {np.allclose(codedpb.sum(0), 0)}")
print(f"main effects mutually orthogonal (XtX = {n_pb}*I): "
      f"{np.allclose(gram, n_pb * np.eye(codedpb.shape[1]))}")

# a readout where three factors dominate; the other eight are inert
rng = np.random.default_rng(19)
true_pb = (
    60
    + 9 * codedpb[:, 3]   # compound_uM
    + 6 * codedpb[:, 6]   # dna_ng
    - 5 * codedpb[:, 1]   # serum_pct
)
y_pb = np.round(true_pb + rng.normal(0, 0.8, size=true_pb.shape), 1)
print("\ny_pb:", list(y_pb))
# fit MAIN EFFECTS ONLY: a saturated PB screen cannot resolve interactions (they are
# partially aliased), so asking for them over-parameterises and dilutes the real effects.
res_pb = fit_ols(pb, y_pb, order=1, interactions=False)
print("\neffects (|effect|, sorted):")
ranked = sorted(res_pb.summary().items(), key=lambda kv: -abs(kv[1][1]))
for name, ce in ranked:
    if name != "Intercept":
        print(f"  {name:>16s}: {ce[1]:+.2f}")

# The figure: the iconic balanced design matrix (left), and the alias-structure
# heatmap (right) -- the partial-aliasing fingerprint that sets PB apart from a
# regular fractional factorial. The right panel reuses doe.plotting.correlation_heatmap.
labels, alias = alias_matrix(pb, interactions=True, absolute=True)
print(f"\nterm-to-term correlations take values: {np.unique(np.round(alias, 4))}")

names = pb.factors.names
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [0.8, 1.4]}
)
ax1.imshow(codedpb, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(n_pb):
    for j in range(len(names)):
        ax1.text(
            j, i, "+" if codedpb[i, j] > 0 else "−",
            ha="center", va="center", fontsize=7,
            color="white" if codedpb[i, j] > 0 else "black",
        )
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=90, fontsize=7)
ax1.set_yticks(range(n_pb))
ax1.set_yticklabels([f"run {i + 1}" for i in range(n_pb)], fontsize=7)
ax1.set_title("PB design: 11 factors in 12 runs\n(red +1, blue −1; every column 6/6 balanced)",
              fontsize=9)

correlation_heatmap(pb, interactions=True, absolute=True, ax=ax2)
ax2.set_title("Alias structure: 11 mains + 55 two-factor interactions\n"
              "(mains mutually orthogonal; each leaks |r| = 1/3 into many 2FIs)", fontsize=9)
save(ax1, "v5_plackett_burman.png")


# --------------------------------------------------------------------------- #
# Vignettes 6-7: CCD, lack-of-fit, contour, diagnostics
# --------------------------------------------------------------------------- #
banner("Vignettes 6-7: central composite design")

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

# Vignette 6 figure: center points reveal curvature a 2-level design can't see.
# Slice the fitted quadratic along DNA at the lipid center: the two corner runs (±1)
# define the straight chord a corners-only design assumes; the center replicates (0)
# sit well above it, and that gap *is* the curvature the lack-of-fit test detects.
gx5, gy5, gz5 = surface_grid(res_ccd, "dna_ng", "lipid_uL", resolution=201)
jcen = int(np.argmin(np.abs(gy5[:, 0] - lipid.decode(np.array([0.0]))[0])))
xline = gx5[0, :]
yline = gz5[jcen, :]
lo_pred, hi_pred, cen_pred = yline[0], yline[-1], yline[len(yline) // 2]
midchord = 0.5 * (lo_pred + hi_pred)
xc = dna.decode(np.array([0.0]))[0]

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(xline, yline, color="tab:blue", lw=2, label="true (fitted) quadratic")
ax.plot([xline[0], xline[-1]], [lo_pred, hi_pred], "--", color="0.5", lw=2,
        label="what corners alone assume: a line")
ax.scatter([xline[0], xline[-1]], [lo_pred, hi_pred], s=95, color="tab:blue",
           edgecolor="white", zorder=5, label="corner runs (coded ±1)")
ax.scatter([xc], [cen_pred], s=130, marker="D", color="tab:orange",
           edgecolor="white", zorder=6, label="center points (coded 0)")
ax.annotate("", xy=(xc, cen_pred), xytext=(xc, midchord),
            arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.8))
ax.text(xc + 8, 0.5 * (cen_pred + midchord), f"curvature\ngap ≈ {cen_pred - midchord:.1f}",
        color="tab:red", fontsize=9, va="center")
ax.set_xlabel("dna_ng")
ax.set_ylabel("predicted % GFP+  (lipid at center)")
ax.set_title("Center points detect curvature a 2-level design can't")
ax.legend(fontsize=8, loc="lower right")
print(f"\ncurvature gap (center vs corner chord): {cen_pred - midchord:.2f}")
save(ax, "v6_curvature.png")

# contour + surface grid optimum
ax = contour_plot(res_ccd, "dna_ng", "lipid_uL")
save(ax, "v7_contour.png")

opt = res_ccd.optimum()
print(f"\noptimum repr: {opt!r}")
print(
    f"predicted optimum: {opt.natural['dna_ng']:.0f} ng DNA, "
    f"{opt.natural['lipid_uL']:.2f} uL lipid -> {opt.response:.1f}% GFP+ "
    f"(at_bound={opt.at_bound})"
)

ax = residuals_vs_fitted(res_ccd)
save(ax, "v9_residuals_vs_fitted.png")
ax = normal_qq(res_ccd)
save(ax, "v9_normal_qq.png")
ax = predicted_vs_actual(res_ccd)
save(ax, "v9_predicted_vs_actual.png")


# --------------------------------------------------------------------------- #
# Vignette 10: ANOVA and significance (reuses the V6 CCD quadratic fit)
# --------------------------------------------------------------------------- #
banner("Vignette 10: ANOVA and significance")

tbl = anova_table(res_ccd, ccd, y_ccd)
print("anova_table(res_ccd, ccd, y_ccd):")
print(tbl.to_string(float_format=lambda v: f"{v:.4g}"))

print("\nper-term (effect, t, p) and 95% CI on the coefficient:")
ci = res_ccd.conf_int(0.95)
for name, eff, t, p, (lo, hi) in zip(
    res_ccd.term_names, res_ccd.effects, res_ccd.t_values, res_ccd.p_values, ci, strict=True
):
    print(f"  {name:>16s}: effect={eff:+7.3f}  t={t:+7.2f}  p={p:.4f}  CI=[{lo:+.3f}, {hi:+.3f}]")

# Vignette 10 figure: a coefficient "forest plot" -- each coefficient as a point with its
# 95% CI as a whisker. An interval that clears the zero line is a significant term made
# visual; the width is how precisely the effect is pinned down.
names8 = [n for n in res_ccd.term_names if n != "Intercept"]
idx8 = [res_ccd.term_names.index(n) for n in names8]
coef8 = res_ccd.coefficients[idx8]
ci8 = ci[idx8]
ypos = np.arange(len(names8))[::-1]
xerr = np.vstack([coef8 - ci8[:, 0], ci8[:, 1] - coef8])
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axvline(0.0, color="crimson", lw=1.3, ls="--", zorder=1)
ax.errorbar(coef8, ypos, xerr=xerr, fmt="none", ecolor="0.4", elinewidth=1.8,
            capsize=5, zorder=2)
ax.scatter(coef8, ypos, s=70, zorder=3,
           color=["tab:blue" if c >= 0 else "tab:red" for c in coef8],
           edgecolor="black", linewidths=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(names8)
ax.set_xlabel("coefficient (coded units), with 95% CI")
ax.set_title("Every term's interval clears zero → all real, not just big")
save(ax, "v10_coefficients.png")


# --------------------------------------------------------------------------- #
# Vignette 11: how much model is too much -- adjusted & predicted R^2
# --------------------------------------------------------------------------- #
banner("Vignette 11: adjusted & predicted R^2")

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

# Figure: the three R^2 flavours side by side for the wrong (linear) vs right
# (quadratic) model. The story is the predicted-R^2 group: the linear bar dives
# below zero while plain R^2 still looks "passable" -- overfitting made visual.
metrics = ["R²", "adjusted R²", "predicted R² (Q²)"]
lin_vals = [res_lin.r_squared, adjusted_r2(res_lin), predicted_r2(res_lin)]
quad_vals = [res_ccd.r_squared, adjusted_r2(res_ccd), predicted_r2(res_ccd)]
xpos = np.arange(len(metrics))
width = 0.38
fig, ax = plt.subplots(figsize=(7.4, 4.8))
bars_lin = ax.bar(
    xpos - width / 2, lin_vals, width, label="linear (flat) model",
    color="#d1495b", edgecolor="black", linewidth=0.6,
)
bars_quad = ax.bar(
    xpos + width / 2, quad_vals, width, label="quadratic model",
    color="#2e8b57", edgecolor="black", linewidth=0.6,
)
ax.axhline(0.0, color="black", linewidth=1.0)
for rects in (bars_lin, bars_quad):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=9,
        )
# call out the collapse: the only bar below zero
ax.annotate(
    "worse than\nguessing the mean",
    xy=(xpos[2] - width / 2, lin_vals[2]),
    xytext=(xpos[2] - width / 2 - 0.05, -0.62),
    ha="center", fontsize=9, color="#d1495b",
    arrowprops=dict(arrowstyle="->", color="#d1495b", linewidth=1.2),
)
ax.set_xticks(xpos)
ax.set_xticklabels(metrics)
ax.set_ylabel("goodness-of-fit score")
ax.set_ylim(-0.85, 1.15)
ax.set_title("R² flatters; predicted R² exposes overfitting")
ax.legend(loc="lower left", framealpha=0.95)
save(ax, "v11_r2_metrics.png")


# --------------------------------------------------------------------------- #
# Vignette 8: Box-Behnken -- a leaner 3-factor response surface
# --------------------------------------------------------------------------- #
banner("Vignette 8: Box-Behnken design")

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
save(ax, "v8_box_behnken_contour.png")


# --------------------------------------------------------------------------- #
# Vignette 12: optimization -- stationary point, canonical analysis, optimum
# --------------------------------------------------------------------------- #
banner("Vignette 12: response-surface optimization")

# happy case: the V6 dome has an interior maximum, recovered analytically
sp = stationary_point(res_ccd)
print("stationary_point(res_ccd):")
print(f"  coded      = {np.array2string(sp.coded, precision=4)}")
print(f"  natural    = {show_nat(sp.natural)}")
print(f"  response   = {sp.response:.2f}")
print(f"  kind       = {sp.kind!r}")
print(f"  eigenvalues= {np.array2string(sp.eigenvalues, precision=3)}")

opt = optimum(res_ccd, maximize=True)
print("\noptimum(res_ccd, maximize=True):")
print(f"  coded={np.array2string(opt.coded, precision=4)}, natural={show_nat(opt.natural)}")
print(f"  response={opt.response:.2f}, at_bound={opt.at_bound}")

# cautionary case: a reporter readout whose model peak lies beyond the tested DNA range
reporter_true = 50 + 14 * cc[:, 0] + 3 * cc[:, 1] - 4 * cc[:, 0] ** 2 - 5 * cc[:, 1] ** 2
rng = np.random.default_rng(5)
y_rep = np.round(reporter_true + rng.normal(0, 1.0, size=reporter_true.shape), 1)
res_rep = fit_ols(ccd, y_rep, model="quadratic")
sp_rep = stationary_point(res_rep)
opt_rep = optimum(res_rep, maximize=True)
print("\nreporter readout (model peak outside the box):")
print(f"  stationary kind={sp_rep.kind!r}, coded={np.array2string(sp_rep.coded, precision=3)}"
      f"  (dna coded {sp_rep.coded[0]:.2f} is outside [-1, 1])")
print(f"  optimum coded={np.array2string(opt_rep.coded, precision=3)}, "
      f"natural={show_nat(opt_rep.natural)}, at_bound={opt_rep.at_bound}")

ax = surface_plot(res_ccd, "dna_ng", "lipid_uL")
save(ax, "v12_surface.png")


# --------------------------------------------------------------------------- #
# Vignette 13: multi-response desirability (Derringer-Suich)
# --------------------------------------------------------------------------- #
banner("Vignette 13: multi-response desirability")

# a second readout on the same CCD: % viable cells, which falls as DNA rises (toxicity)
viab_true = 80 - 15 * cc[:, 0] - 2 * cc[:, 1] - 3 * cc[:, 0] ** 2
rng = np.random.default_rng(3)
y_viab = np.round(viab_true + rng.normal(0, 1.0, size=viab_true.shape), 1)
res_viab = fit_ols(ccd, y_viab, model="quadratic")
print("viability fit summary():")
for k, (c, e) in res_viab.summary().items():
    print(f"  {k!r}: (coef={c:.4g}, effect={e:.4g})")

goals = [
    ResponseGoal(res_ccd, goal="max", low=40.0, high=70.0),   # maximise % GFP+
    ResponseGoal(res_viab, goal="max", low=50.0, high=90.0),  # maximise % viable
]
des = desirability(goals)
print("\ndesirability([GFP max, viability max]):")
print(f"  coded     = {np.array2string(des.coded, precision=4)}")
print(f"  natural   = {show_nat(des.natural)}")
print(f"  responses = (GFP {des.responses[0]:.1f}%, viability {des.responses[1]:.1f}%)")
print(f"  individual d = {np.array2string(des.individual, precision=3)}")
print(f"  overall D = {des.overall:.3f}")

# for contrast: optimising GFP alone pushes to high DNA, where viability is worse.
# read the viability surface at the GFP-only optimum to show the trade-off the
# desirability solution avoids.
opt_gfp = optimum(res_ccd, maximize=True)
gx, gy, gz_v = surface_grid(res_viab, "dna_ng", "lipid_uL", resolution=201)
ii = int(np.argmin(np.abs(gx[0, :] - opt_gfp.natural["dna_ng"])))
jj = int(np.argmin(np.abs(gy[:, 0] - opt_gfp.natural["lipid_uL"])))
print(
    f"\nGFP-only optimum (dna_ng={opt_gfp.natural['dna_ng']:.0f}, GFP {opt_gfp.response:.1f}%): "
    f"viability there is only {gz_v[jj, ii]:.0f}% -- the desirability point trades a little "
    f"GFP for much better viability."
)

# Vignette 13 figure: the overall desirability D as a surface, with the compromise it
# strikes (gold star) sitting away from the GFP-only optimum (red X). D combines both
# readouts, so its peak lands where *both* stay acceptable, not where GFP alone is highest.
gx12, gy12, z_gfp = surface_grid(res_ccd, "dna_ng", "lipid_uL", resolution=201)
_, _, z_viab = surface_grid(res_viab, "dna_ng", "lipid_uL", resolution=201)
d_gfp = np.vectorize(goals[0].desirability)(z_gfp)
d_viab = np.vectorize(goals[1].desirability)(z_viab)
d_grid = np.sqrt(np.clip(d_gfp, 0.0, None) * np.clip(d_viab, 0.0, None))
fig, ax = plt.subplots(figsize=(7.4, 5.4))
cf = ax.contourf(gx12, gy12, d_grid, levels=np.linspace(0.0, float(d_grid.max()), 13),
                 cmap="viridis")
fig.colorbar(cf, ax=ax, label="overall desirability D")
ax.contour(gx12, gy12, d_grid, levels=8, colors="white", linewidths=0.4, alpha=0.5)
ax.scatter([des.natural["dna_ng"]], [des.natural["lipid_uL"]], s=260, marker="*",
           color="gold", edgecolors="black", linewidths=1.2, zorder=6,
           label=f"desirability optimum (D={des.overall:.2f})")
ax.scatter([opt_gfp.natural["dna_ng"]], [opt_gfp.natural["lipid_uL"]], s=130, marker="X",
           color="crimson", edgecolors="white", linewidths=1.2, zorder=6,
           label="GFP-only optimum")
ax.set_xlabel("dna_ng")
ax.set_ylabel("lipid_uL")
ax.set_title("Desirability balances %GFP+ against viability")
ax.legend(loc="lower left", fontsize=8, facecolor="#1a1a1a", edgecolor="white",
          framealpha=0.85, labelcolor="white")
save(ax, "v13_desirability.png")


# --------------------------------------------------------------------------- #
# Vignette 14: randomise run order
# --------------------------------------------------------------------------- #
banner("Vignette 14: randomise run order")

plate_order = ccd.randomize(seed=42)
print("plate_order.runs.head():")
print(plate_order.runs.head())

# Vignette 14 figure: why run order matters. A lurking plate/time drift (grey dashed) runs
# across the pipetting order. If you run the plate SORTED by DNA, DNA's level marches with
# that drift and the two are confounded -- the drift masquerades as a DNA effect. Randomising
# the order decorrelates them, so the drift becomes noise spread across all factors.
cc_dna = ccd.coded().to_numpy(dtype=float)[:, 0]
n13 = len(cc_dna)
order_sorted = np.argsort(cc_dna, kind="stable")
order_rand = plate_order.runs["std_order"].to_numpy()
run_idx = np.arange(1, n13 + 1)
drift = np.linspace(-1.0, 1.0, n13)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, order, title in [
    (axL, order_sorted, "Sorted run order (DNA ascending)"),
    (axR, order_rand, "Randomised run order (seed=42)"),
]:
    dna_lvl = cc_dna[order]
    r = _corr(run_idx.astype(float), dna_lvl)
    ax.bar(run_idx, dna_lvl, width=0.62,
           color=["tab:red" if v > 0 else "tab:blue" if v < 0 else "0.7" for v in dna_lvl],
           alpha=0.85, zorder=2, label="DNA coded level")
    ax.plot(run_idx, drift, color="0.35", lw=2, ls="--", zorder=3, label="plate/time drift")
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel("pipetting order (run #)")
    ax.set_ylim(-1.25, 1.25)
    ax.set_title(f"{title}\ncorr(drift, DNA) = {r:+.2f}", fontsize=10)
axL.set_ylabel("coded level / normalised drift")
axL.legend(loc="lower right", fontsize=8)
r_sorted = _corr(run_idx.astype(float), cc_dna[order_sorted])
r_rand = _corr(run_idx.astype(float), cc_dna[order_rand])
print(f"\nconfounding with a linear drift: corr sorted={r_sorted:+.2f}, randomised={r_rand:+.2f}")
save(axL, "v14_randomization.png")


# --------------------------------------------------------------------------- #
# Vignette 15: interactive HTML view of the design
# --------------------------------------------------------------------------- #
banner("Vignette 15: interactive HTML view of the design")

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
# the randomised CCD doubles as a bench run sheet: 'run' is the pipetting order and
# 'std_order' maps each well back to its design row for re-joining readouts.
plate_order.name = "Transfection CCD - randomised run sheet"
html = to_html(plate_order, path=DOCS / "example_design.html")
print(f"wrote docs/example_design.html ({len(html)} bytes)")


# --------------------------------------------------------------------------- #
# Vignette 16: alias-structure heatmap (correlation_heatmap)
# --------------------------------------------------------------------------- #
banner("Vignette 16: alias-structure heatmap")

# a 2^(4-1) fraction with short factor names so the term labels match the generator
demo = fractional_factorial(
    [ContinuousFactor(c, 0.0, 1.0) for c in "ABCD"], generators=["D=ABC"]
)
labels15, alias15 = alias_matrix(demo, interactions=True)
idx15 = {name: i for i, name in enumerate(labels15)}
print(f"labels: {labels15}")
print("confounded two-factor-interaction pairs (|r| = 1):")
for left, right in [("A:B", "C:D"), ("A:C", "B:D"), ("A:D", "B:C")]:
    print(f"  {left} = {right}: r = {alias15[idx15[left], idx15[right]]:+.0f}")

ax = correlation_heatmap(demo, interactions=True)
save(ax, "v16_alias.png")


# --------------------------------------------------------------------------- #
# Vignette 17: design diagnostics before running the plate
# --------------------------------------------------------------------------- #
banner("Vignette 17: design diagnostics")

diag = efficiency(ccd, order=2, interactions=True)
print("efficiency(ccd, order=2, interactions=True):")
print(f"  D = {diag.d:.3f}")
print(f"  A = {diag.a:.3f}")
print(f"  G = {diag.g:.3f}")
print(f"  I = {diag.i:.3f}")
print(f"condition_number(res_ccd.model_matrix) = {condition_number(res_ccd.model_matrix):.2f}")
print("\nVIFs:")
for name, value in vif(res_ccd.model_matrix, term_names=res_ccd.term_names).items():
    print(f"  {name:>16s}: {value:.2f}")

ax = leverage_plot(res_ccd)
save(ax, "v17_leverage.png")


# --------------------------------------------------------------------------- #
# Vignette 18: computer-generated optimal designs
# --------------------------------------------------------------------------- #
banner("Vignette 18: optimal and augmented designs")

region = candidate_grid([dna, lipid], levels=5)
d_design = d_optimal(
    [dna, lipid], n_runs=8, model="quadratic", region=region, seed=1, n_restarts=50
)
i_design = i_optimal(
    [dna, lipid], n_runs=8, model="quadratic", region=region, seed=1, n_restarts=50
)
d_eff = efficiency(d_design, order=2, interactions=True, region=region)
i_eff = efficiency(i_design, order=2, interactions=True, region=region)
print("D-optimal 8-run quadratic design:")
print(d_design.runs)
print("I-optimal 8-run quadratic design:")
print(i_design.runs)
print(
    "\nshared-region efficiencies:\n"
    f"  D-optimal: D={d_eff.d:.3f}, I={d_eff.i:.3f}\n"
    f"  I-optimal: D={i_eff.d:.3f}, I={i_eff.i:.3f}"
)

fig, ax = plt.subplots(figsize=(6.2, 4.8))
region_nat_x = dna.decode(region[:, 0])
region_nat_y = lipid.decode(region[:, 1])
ax.scatter(region_nat_x, region_nat_y, s=25, color="0.85", label="candidate grid")
d_coded = d_design.coded().to_numpy(dtype=float)
i_coded = i_design.coded().to_numpy(dtype=float)
ax.scatter(
    dna.decode(d_coded[:, 0]),
    lipid.decode(d_coded[:, 1]),
    marker="s",
    s=80,
    facecolors="none",
    edgecolors="tab:blue",
    linewidths=1.8,
    label="D-optimal runs",
)
ax.scatter(
    dna.decode(i_coded[:, 0]),
    lipid.decode(i_coded[:, 1]),
    marker="^",
    s=80,
    facecolors="none",
    edgecolors="tab:orange",
    linewidths=1.8,
    label="I-optimal runs",
)
ax.set_xlabel("dna_ng")
ax.set_ylabel("lipid_uL")
ax.set_title("Candidate grid with D- and I-optimal 8-run choices")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
save(ax, "v18_optimal_designs.png")

augmented = augment(full_factorial([dna, lipid]), n_runs=4, model="quadratic", seed=2)
print("\naugment(full_factorial([dna, lipid]), n_runs=4):")
print(augmented.runs)
print(f"point_types = {augmented.point_types}")

reagent = CategoricalFactor("reagent", ("PEI", "Lipo", "FuGENE"))
mixed = d_optimal([dna, reagent], n_runs=6, model="linear", seed=4)
print("\nmixed continuous/categorical D-optimal design:")
print(mixed.runs)


# --------------------------------------------------------------------------- #
# Vignette 19: space-filling designs (LHS, Sobol', Halton) + coverage metrics
# --------------------------------------------------------------------------- #
banner("Vignette 19: space-filling designs")


def design_from_coded(coded: np.ndarray, factors: list[ContinuousFactor]) -> Design:
    """Wrap a coded [-1, +1]^k point cloud as a Design (for discrepancy/plots)."""
    data = {f.name: f.decode(coded[:, j]) for j, f in enumerate(factors)}
    return Design(pd.DataFrame(data), FactorSet(factors))


sf_dna = ContinuousFactor("dna_ng", 100, 500, units="ng/well")
sf_lipid = ContinuousFactor("lipid_uL", 0.5, 2.5, units="uL/well")
sf_factors = [sf_dna, sf_lipid]
n = 16

# four ways to place 16 points over the same box
rng = np.random.default_rng(0)
iid16 = design_from_coded(rng.uniform(-1.0, 1.0, size=(n, 2)), sf_factors)
g = np.linspace(-1.0, 1.0, 4)
gx, gy = np.meshgrid(g, g)
grid16 = design_from_coded(np.column_stack([gx.ravel(), gy.ravel()]), sf_factors)
lhs16 = latin_hypercube(sf_factors, n_runs=n, seed=0)
sobol16 = sobol(sf_factors, n_runs=n, seed=0)

panels = [
    ("Random (i.i.d.)", iid16),
    ("Grid (4x4)", grid16),
    ("Latin hypercube", lhs16),
    ("Sobol'", sobol16),
]
fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6))
for ax, (title, d) in zip(axes.ravel(), panels, strict=True):
    r = d.runs
    ax.scatter(r["dna_ng"], r["lipid_uL"], s=45, color="tab:blue", edgecolor="white", zorder=3)
    ax.set_title(f"{title}\ndiscrepancy = {discrepancy(d):.4f}", fontsize=10)
    ax.set_xlabel("dna_ng")
    ax.set_ylabel("lipid_uL")
    ax.set_xlim(90, 510)
    ax.set_ylim(0.45, 2.55)
save(axes[0, 0], "v19_spacefilling_compare.png")

print("coverage of the four 16-run designs over [dna_ng, lipid_uL]:")
for label, d in panels:
    print(
        f"  {label:>16s}: discrepancy={discrepancy(d):.4f}, "
        f"maximin_distance={maximin_distance(d):.4f}"
    )

# stratification diagram: an 8-run LHS with stratum gridlines + marginal rug ticks
lhs8 = latin_hypercube(sf_factors, n_runs=8, seed=1)
print("\nlhs8 = latin_hypercube([dna_ng, lipid_uL], n_runs=8, seed=1)")
print("lhs8.runs:")
print(lhs8.runs)
print("lhs8.meta:", lhs8.meta)

r8 = lhs8.runs
fig, ax = plt.subplots(figsize=(6.4, 5.8))
for e in np.linspace(100, 500, 9):
    ax.axvline(e, color="0.85", lw=0.8, zorder=0)
for e in np.linspace(0.5, 2.5, 9):
    ax.axhline(e, color="0.85", lw=0.8, zorder=0)
ax.scatter(r8["dna_ng"], r8["lipid_uL"], s=60, color="tab:blue", edgecolor="white", zorder=3)
ax.plot(
    r8["dna_ng"], np.full(8, 0.5), "|", color="tab:red",
    markersize=14, markeredgewidth=1.8, clip_on=False, zorder=4,
)
ax.plot(
    np.full(8, 100), r8["lipid_uL"], "_", color="tab:red",
    markersize=14, markeredgewidth=1.8, clip_on=False, zorder=4,
)
ax.set_xlim(100, 500)
ax.set_ylim(0.5, 2.5)
ax.set_xlabel("dna_ng")
ax.set_ylabel("lipid_uL")
ax.set_title("Latin hypercube: exactly one point per stratum, on every axis (n = 8)")
save(ax, "v19_lhs_stratification.png")

# sobol' rejects non-power-of-two run counts
try:
    sobol(sf_factors, n_runs=20)
except ValueError as exc:
    print("\nsobol([dna_ng, lipid_uL], n_runs=20) ->", exc)

# discrepancy convergence: random vs LHS vs Halton vs Sobol', averaged over seeds
conv_factors = [ContinuousFactor(name, 0.0, 1.0) for name in ("x1", "x2", "x3")]
ns = [8, 16, 32, 64, 128]
n_seed = 15
rand_disc, lhs_disc, hal_disc, sob_disc = [], [], [], []
for nn in ns:
    rd, ld, hd, sd = [], [], [], []
    for s in range(n_seed):
        cloud = np.random.default_rng(1000 + s).uniform(-1.0, 1.0, size=(nn, 3))
        rd.append(discrepancy(design_from_coded(cloud, conv_factors)))
        ld.append(discrepancy(latin_hypercube(conv_factors, n_runs=nn, criterion=None, seed=s)))
        hd.append(discrepancy(halton(conv_factors, n_runs=nn, seed=s)))
        sd.append(discrepancy(sobol(conv_factors, n_runs=nn, seed=s)))
    rand_disc.append(float(np.mean(rd)))
    lhs_disc.append(float(np.mean(ld)))
    hal_disc.append(float(np.mean(hd)))
    sob_disc.append(float(np.mean(sd)))

print("\ndiscrepancy vs n (mean over 15 seeds, 3 factors):")
print(f"  {'n':>4} {'random':>9} {'lhs':>9} {'halton':>9} {'sobol':>9}")
for i, nn in enumerate(ns):
    print(
        f"  {nn:>4} {rand_disc[i]:>9.4f} {lhs_disc[i]:>9.4f} "
        f"{hal_disc[i]:>9.4f} {sob_disc[i]:>9.4f}"
    )

fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.loglog(ns, rand_disc, "o-", color="0.5", label="random (i.i.d.)")
ax.loglog(ns, lhs_disc, "s-", color="tab:green", label="Latin hypercube")
ax.loglog(ns, hal_disc, "^-", color="tab:orange", label="Halton")
ax.loglog(ns, sob_disc, "D-", color="tab:blue", label="Sobol'")
ax.set_xlabel("number of runs")
ax.set_ylabel("centered discrepancy (lower = more uniform)")
ax.set_title("Coverage improves faster for low-discrepancy sequences (3 factors)")
ax.set_xticks(ns)
ax.set_xticklabels([str(nn) for nn in ns])
ax.legend()
save(ax, "v19_discrepancy_convergence.png")


# --------------------------------------------------------------------------- #
# Vignette 20: mixture designs -- a three-solvent formulation
# --------------------------------------------------------------------------- #
banner("Vignette 20: mixture designs")

# Three co-solvents whose proportions must sum to 1: the response (a coating's gloss)
# depends only on the *blend*, not the absolute amount. This is a mixture problem.
solvents = [
    MixtureFactor("water"),
    MixtureFactor("ethanol"),
    MixtureFactor("acetone"),
]

lattice = simplex_lattice(solvents, degree=2)
print("simplex_lattice(solvents, degree=2).runs:")
print(lattice.runs)
print("\npoint_types:", lattice.point_types)

centroid = simplex_centroid(solvents)
print(f"\nsimplex_centroid: {centroid.n_runs} runs (2^3 - 1)")

# A synthetic ground-truth blending surface (Scheffe quadratic): ethanol/acetone
# blends synergise, water antagonises. Measure gloss on the centroid design (with the
# overall centroid replicated so a pure-error / lack-of-fit read is possible).
def true_gloss(w: Any, e: Any, a: Any) -> Any:
    return 40 * w + 55 * e + 60 * a + 40 * e * a - 25 * w * e


meas = simplex_centroid(solvents).replicate(1)
props = meas.runs.to_numpy(dtype=float)
rng = np.random.default_rng(20)
gloss = true_gloss(props[:, 0], props[:, 1], props[:, 2]) + rng.normal(0, 0.4, len(props))

result = fit_ols(meas, gloss, model="scheffe-quadratic")
print("\nfit_ols(meas, gloss, model='scheffe-quadratic'):")
for name, coef in zip(result.term_names, result.coefficients, strict=True):
    print(f"  {name!r}: {coef:+.2f}")
print(f"\nR^2 = {result.r_squared:.4f}")
print("model matrix has no intercept column:", "Intercept" not in result.term_names)

# Mixture ANOVA: the k linear terms collapse into one "Linear blending" row (k-1 df),
# then a 1-df row per cross product -- the textbook convention. Confirms which
# synergy/antagonism terms are statistically real, not just large.
mix_tbl = anova_table(result, meas, gloss)
print("\nanova_table(result, meas, gloss):")
print(mix_tbl.round(3))

# Read the best blend straight off the fitted surface: argmax over the ternary grid.
gx, gy, gz, gpts = ternary_grid(result, resolution=200)
best = gpts[int(np.argmax(gz))]
print(
    f"\nbest grid blend: water={best[0]:.2f}, ethanol={best[1]:.2f}, "
    f"acetone={best[2]:.2f} -> gloss={gz.max():.1f}"
)

# a constrained region: water must stay >= 30%, acetone <= 50% -- the recipes above
# assume the full simplex, so a bounded region goes to extreme_vertices.
constrained = [
    MixtureFactor("water", low=0.30, high=1.0),
    MixtureFactor("ethanol", low=0.0, high=0.60),
    MixtureFactor("acetone", low=0.0, high=0.50),
]
ev = extreme_vertices(constrained)
print("\nextreme_vertices(constrained).runs:")
print(ev.runs)

# D-optimal mixture design for an odd run budget: the Phase 3 engine over the mixture
# candidate set. Every row still sums to 1 (whole-candidate exchanges preserve it).
region = mixture_candidates(solvents, resolution=4)
dopt = d_optimal(solvents, n_runs=7, model="quadratic", region=region, n_restarts=8, seed=0)
row_sums = dopt.runs.to_numpy(dtype=float).sum(axis=1)
print(f"\nd_optimal mixture (7 runs): row sums all 1.0 -> {np.allclose(row_sums, 1.0)}")

ax = ternary_contour(result, meas)
save(ax, "v20_ternary_contour.png")

print("\nDONE")
