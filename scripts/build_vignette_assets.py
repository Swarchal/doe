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
import pandas as pd

from doe import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    FactorSet,
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
    fit_ols,
    fractional_factorial,
    full_factorial,
    halton,
    i_optimal,
    latin_hypercube,
    maximin_distance,
    optimum,
    plackett_burman,
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
# Vignette 4b: Plackett-Burman -- the leanest main-effect screen
# --------------------------------------------------------------------------- #
banner("Vignette 4b: Plackett-Burman screening")

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
save(ax1, "v4b_plackett_burman.png")


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

opt = res_ccd.optimum()
print(f"\noptimum repr: {opt!r}")
print(
    f"predicted optimum: {opt.natural['dna_ng']:.0f} ng DNA, "
    f"{opt.natural['lipid_uL']:.2f} uL lipid -> {opt.response:.1f}% GFP+ "
    f"(at_bound={opt.at_bound})"
)

ax = residuals_vs_fitted(res_ccd)
save(ax, "v7_residuals_vs_fitted.png")
ax = normal_qq(res_ccd)
save(ax, "v7_normal_qq.png")
ax = predicted_vs_actual(res_ccd)
save(ax, "v7_predicted_vs_actual.png")


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
# Vignette 11: optimization -- stationary point, canonical analysis, optimum
# --------------------------------------------------------------------------- #
banner("Vignette 11: response-surface optimization")

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
save(ax, "v11_surface.png")


# --------------------------------------------------------------------------- #
# Vignette 12: multi-response desirability (Derringer-Suich)
# --------------------------------------------------------------------------- #
banner("Vignette 12: multi-response desirability")

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


# --------------------------------------------------------------------------- #
# Vignette 13: randomise run order
# --------------------------------------------------------------------------- #
banner("Vignette 13: randomise run order")

plate_order = ccd.randomize(seed=42)
print("plate_order.runs.head():")
print(plate_order.runs.head())


# --------------------------------------------------------------------------- #
# Vignette 14: interactive HTML view of the design
# --------------------------------------------------------------------------- #
banner("Vignette 14: interactive HTML view of the design")

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
# the randomised CCD doubles as a bench run sheet: 'run' is the pipetting order and
# 'std_order' maps each well back to its design row for re-joining readouts.
plate_order.name = "Transfection CCD - randomised run sheet"
html = to_html(plate_order, path=DOCS / "example_design.html")
print(f"wrote docs/example_design.html ({len(html)} bytes)")


# --------------------------------------------------------------------------- #
# Vignette 15: alias-structure heatmap (correlation_heatmap)
# --------------------------------------------------------------------------- #
banner("Vignette 15: alias-structure heatmap")

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
save(ax, "v15_alias.png")


# --------------------------------------------------------------------------- #
# Vignette 16: design diagnostics before running the plate
# --------------------------------------------------------------------------- #
banner("Vignette 16: design diagnostics")

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
save(ax, "v16_leverage.png")


# --------------------------------------------------------------------------- #
# Vignette 17: computer-generated optimal designs
# --------------------------------------------------------------------------- #
banner("Vignette 17: optimal and augmented designs")

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
ax.legend(loc="best")
save(ax, "v17_optimal_designs.png")

augmented = augment(full_factorial([dna, lipid]), n_runs=4, model="quadratic", seed=2)
print("\naugment(full_factorial([dna, lipid]), n_runs=4):")
print(augmented.runs)
print(f"point_types = {augmented.point_types}")

reagent = CategoricalFactor("reagent", ("PEI", "Lipo", "FuGENE"))
mixed = d_optimal([dna, reagent], n_runs=6, model="linear", seed=4)
print("\nmixed continuous/categorical D-optimal design:")
print(mixed.runs)


# --------------------------------------------------------------------------- #
# Vignette 18: space-filling designs (LHS, Sobol', Halton) + coverage metrics
# --------------------------------------------------------------------------- #
banner("Vignette 18: space-filling designs")


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
save(axes[0, 0], "v18_spacefilling_compare.png")

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
save(ax, "v18_lhs_stratification.png")

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
save(ax, "v18_discrepancy_convergence.png")

print("\nDONE")
