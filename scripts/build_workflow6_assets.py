"""Run the docs/WORKFLOW6.md walkthrough, capture real outputs and save its figures.

This is the provenance for the outputs/figures embedded in docs/WORKFLOW6.md: it reproduces
the constrained-optimal-design example verbatim (same factors, same seeds), prints every
console block for transcription, and writes the figures to docs/img/ with a ``wf6_`` prefix.
Run with: uv run python scripts/build_workflow6_assets.py

The story is deliberately different from the other walkthroughs. WORKFLOW/WORKFLOW3 map a box
region with a named recipe (central composite); WORKFLOW5 lives on the simplex. Here the
situation fits *no* named recipe at all: one factor is categorical (a catalyst choice, which
the response-surface generators reject outright), a safety constraint carves an irregular
corner out of the box, and the run budget is tight and odd. That is exactly what the
coordinate-exchange engine (d_optimal/i_optimal) is for -- it *builds* a run set for the model,
budget, and feasible region you actually have. The response is a synthetic-but-realistic
surface so the doc is fully reproducible.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from doe import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    FactorSet,
    box_behnken,
    candidate_grid,
    condition_number,
    correlation_matrix,
    d_optimal,
    efficiency,
    fit_ols,
    vif,
)
from doe.analysis.model import build_model_matrix

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)


def save(fig: object, name: str) -> None:
    fig.tight_layout()  # type: ignore[attr-defined]
    fig.savefig(IMG / name, dpi=110, bbox_inches="tight")  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[arg-type]
    print(f"  wrote {name}")


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# The safety constraint, expressed in coded units: a run may be hot OR co-solvent-rich,
# but not both. This carves the high-temperature/high-co-solvent corner out of the box.
def feasible_mask(temp_c: np.ndarray, cosolv_c: np.ndarray) -> np.ndarray:
    return temp_c + cosolv_c <= 0.5 + 1e-9


def decode_candidates(coded: np.ndarray, factors: list) -> pd.DataFrame:
    """Turn coded candidate rows into a natural-unit runs frame (continuous + categorical)."""
    cols: dict[str, object] = {}
    for j, f in enumerate(factors):
        if isinstance(f, ContinuousFactor):
            cols[f.name] = f.decode(coded[:, j])
        else:  # categorical: snap to the nearest discrete contrast level
            levels = np.linspace(-1.0, 1.0, len(f.levels))
            idx = np.abs(coded[:, [j]] - levels[None, :]).argmin(axis=1)
            cols[f.name] = [f.levels[int(i)] for i in idx]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# 1. A design problem no named recipe fits
# --------------------------------------------------------------------------- #
banner("Section 1: the factors -- one categorical, two continuous")

factors = [
    CategoricalFactor("catalyst", levels=("Pd", "Ni", "Cu")),
    ContinuousFactor("temperature", low=60, high=100, units="C"),
    ContinuousFactor("cosolvent", low=5, high=25, units="%"),
]
for f in factors:
    if isinstance(f, ContinuousFactor):
        print(f"  {f.name:12s} continuous [{f.low}, {f.high}] {f.units}")
    else:
        print(f"  {f.name:12s} categorical {list(f.levels)}")

# The response-surface recipes cannot even accept these factors.
try:
    box_behnken(factors)
except (ValueError, TypeError) as exc:
    print(f"\nbox_behnken(factors) -> {type(exc).__name__}:\n  {exc}")


# --------------------------------------------------------------------------- #
# 2. The region is not a box -- a safety constraint carves out a corner
# --------------------------------------------------------------------------- #
banner("Section 2: the feasible candidate region")

grid = candidate_grid(factors, levels=5)  # catalyst(3) x temp(5) x cosolvent(5) = 75
# columns: 0 catalyst, 1 temperature, 2 cosolvent
feasible = grid[feasible_mask(grid[:, 1], grid[:, 2])]
print(f"full grid candidates:      {len(grid)}")
print(f"feasible candidates:       {len(feasible)}  (constraint temp+cosolvent <= 0.5, coded)")
print(f"removed by the constraint: {len(grid) - len(feasible)}")

# Figure A: the feasible region in the (temperature, cosolvent) plane (identical for each
# catalyst -- the constraint is on the continuous factors only).
plane = np.unique(grid[:, 1:3], axis=0)
feas = plane[feasible_mask(plane[:, 0], plane[:, 1])]
infeas = plane[~feasible_mask(plane[:, 0], plane[:, 1])]
temp = factors[1]
cosolv = factors[2]

fig, ax = plt.subplots(figsize=(6.0, 5.4))
ax.add_patch(Rectangle((60, 5), 40, 20, fill=False, ec="0.6", lw=1.2))
# shade the forbidden corner: temp_c + cosolv_c > 0.5
tt = np.linspace(60, 100, 200)
cc = np.linspace(5, 25, 200)
TT, CC = np.meshgrid(tt, cc)
forbidden = (TT - 80) / 20 + (CC - 15) / 10 > 0.5
ax.contourf(TT, CC, forbidden, levels=[0.5, 1.5], colors=["tab:red"], alpha=0.12)
ax.scatter(temp.decode(feas[:, 0]), cosolv.decode(feas[:, 1]), s=45, color="tab:blue",
           edgecolors="white", linewidths=0.8, zorder=4, label="feasible candidate")
ax.scatter(temp.decode(infeas[:, 0]), cosolv.decode(infeas[:, 1]), s=45, marker="x",
           color="tab:red", zorder=4, label="excluded (unsafe)")
ax.set_xlabel("temperature (C)")
ax.set_ylabel("cosolvent (%)")
ax.set_title("Feasible region: hot OR co-solvent-rich, not both", pad=12)
ax.legend(loc="lower left", fontsize=8)
save(fig, "wf6_region.png")


# --------------------------------------------------------------------------- #
# 3. Let the coordinate-exchange engine choose the runs
# --------------------------------------------------------------------------- #
banner("Section 3: a 15-run D-optimal design over the feasible region")

N_RUNS = 15
d_design = d_optimal(
    factors, n_runs=N_RUNS, model="quadratic", region=feasible, seed=20260711, n_restarts=40
)
print(f"chosen {d_design.n_runs} runs; d_efficiency = {d_design.meta['d_efficiency']:.3f}, "
      f"seed = {d_design.meta['seed']}")
print("\nruns (natural units), sorted for readability:")
shown = d_design.runs.sort_values(["catalyst", "temperature", "cosolvent"]).round(1)
print(shown.to_string(index=False))
balance = d_design.runs["catalyst"].value_counts().sort_index()
print("\ncatalyst balance:", {k: int(v) for k, v in balance.items()})

# every chosen run really is inside the feasible region
coded = d_design.coded().to_numpy()
print("all runs feasible:", bool(np.all(feasible_mask(coded[:, 1], coded[:, 2]))))


# --------------------------------------------------------------------------- #
# 4. Prove it: diagnostics vs a naive feasible design
# --------------------------------------------------------------------------- #
banner("Section 4: diagnostics -- D-optimal vs a naive design")

# A naive design: just grab 15 allowed points at random (no thought for the model).
rng = np.random.default_rng(7)
pick = rng.choice(len(feasible), size=N_RUNS, replace=False)
naive = Design(decode_candidates(feasible[pick], factors), FactorSet(factors), name="naive")


def report(name: str, design: Design) -> None:
    eff = efficiency(design, order=2, interactions=True, region=feasible)
    mm = build_model_matrix(design, order=2, interactions=True)
    max_vif = vif(mm.X, term_names=mm.term_names).drop("Intercept", errors="ignore").max()
    cond = condition_number(mm.X)
    print(f"  {name:22s} D={eff.d:.3f}  A={eff.a:.3f}  G={eff.g:.3f}  I={eff.i:.3f}  "
          f"max VIF={max_vif:5.2f}  cond={cond:6.1f}")


print("efficiencies (1.0 = orthogonal ideal), max VIF and condition number, quadratic model:")
report("naive (random)", naive)
report("D-optimal", d_design)

# Figure B: the D-optimal design's alias structure -- near-zero off-diagonals = clean estimates.
mm = build_model_matrix(d_design, order=2, interactions=True)
corr = correlation_matrix(mm.X, mm.term_names)
fig, ax = plt.subplots(figsize=(6.6, 5.8))
im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.index, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
ax.set_title("D-optimal alias structure (off-diagonals near 0)", pad=12)
save(fig, "wf6_alias.png")


# --------------------------------------------------------------------------- #
# 5. Fit the model and find the best feasible operating point
# --------------------------------------------------------------------------- #
banner("Section 5: fit the quadratic and find the best feasible setting")

# "Truth": a catalyst-dependent quadratic surface. Ni is the best catalyst; the payoff climbs
# with both temperature and co-solvent and they reinforce each other -- so the model's peak is
# pushed into the forbidden hot/rich corner, and the best *feasible* run sits on the safety line.
CAT_OFFSET = {"Pd": 3.0, "Ni": 8.0, "Cu": 0.0}
CAT_TEMP = {"Pd": 0.0, "Ni": 4.0, "Cu": -2.0}  # Ni benefits most from heat


def bench_yield(design: Design, rng: np.random.Generator) -> np.ndarray:
    coded = design.coded().to_numpy()
    cat = design.runs["catalyst"].to_numpy()
    t, c = coded[:, 1], coded[:, 2]
    base = np.array([CAT_OFFSET[k] for k in cat])
    ct = np.array([CAT_TEMP[k] for k in cat])
    y = (
        70.0 + base
        + (5.0 + ct) * t + 6.0 * c
        - 7.0 * t**2 - 6.0 * c**2
        + 4.0 * t * c
    )
    return np.asarray(y + rng.normal(0.0, 1.0, design.n_runs))


rng = np.random.default_rng(20260711)
measured = d_design.with_response("yield_pct", bench_yield(d_design, rng))

fit = fit_ols(measured, "yield_pct", model="quadratic")
print(f"R2={fit.r_squared:.3f}  adjR2={fit.adjusted_r2():.3f}  predR2={fit.predicted_r2():.3f}")
print(fit.summary().round(2))

# The surface optimizer can't help here: it needs all-continuous factors and cannot see the
# constraint. So search the feasible region directly, exactly as the mixture walkthrough does.
try:
    fit.optimum(maximize=True)
except (TypeError, ValueError) as exc:
    print(f"\nfit.optimum(maximize=True) -> {type(exc).__name__}:\n  {exc}")

# Dense feasible search: every catalyst over a fine (temperature, cosolvent) grid.
tg = np.linspace(60, 100, 81)
cg = np.linspace(5, 25, 81)
TG, CG = np.meshgrid(tg, cg)
tc = (TG.ravel() - 80) / 20
cc_ = (CG.ravel() - 15) / 10
keep = feasible_mask(tc, cc_)
rows = []
for cat in ["Pd", "Ni", "Cu"]:
    df = pd.DataFrame({
        "catalyst": cat,
        "temperature": TG.ravel()[keep],
        "cosolvent": CG.ravel()[keep],
    })
    df["pred"] = np.asarray(fit.predict(df), dtype=float)
    rows.append(df)
search = pd.concat(rows, ignore_index=True)
best = search.loc[search["pred"].idxmax()]
print(f"\nsearched {len(search)} feasible points across all three catalysts")
print(f"best feasible blend: catalyst={best['catalyst']}, "
      f"temperature={best['temperature']:.1f} C, cosolvent={best['cosolvent']:.1f} %")
print(f"predicted yield = {best['pred']:.1f}%")

best_row = pd.DataFrame([{
    "catalyst": best["catalyst"],
    "temperature": float(best["temperature"]),
    "cosolvent": float(best["cosolvent"]),
}])
print("\n95% prediction interval at the best feasible setting:")
print(fit.predict(best_row, interval="prediction").round(1))
on_line = abs((best["temperature"] - 80) / 20 + (best["cosolvent"] - 15) / 10 - 0.5) < 0.05
print(f"best point sits on the safety line: {on_line}")

# Figure C: the fitted surface for the winning catalyst, feasible region + optimum starred.
best_cat = str(best["catalyst"])
df_full = pd.DataFrame({"catalyst": best_cat, "temperature": TG.ravel(), "cosolvent": CG.ravel()})
pred_full = np.asarray(fit.predict(df_full), dtype=float).reshape(TG.shape)
forbidden = (TG - 80) / 20 + (CG - 15) / 10 > 0.5
pred_masked = np.ma.array(pred_full, mask=forbidden)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
cs = ax.contourf(TG, CG, pred_masked, levels=14, cmap="viridis")
ax.contourf(TG, CG, forbidden, levels=[0.5, 1.5], colors=["0.85"], alpha=1.0)
fig.colorbar(cs, ax=ax, label="predicted yield (%)")
runs_cat = measured.runs[measured.runs["catalyst"] == best_cat]
ax.scatter(runs_cat["temperature"], runs_cat["cosolvent"], s=55, color="white",
           edgecolors="black", linewidths=1.0, zorder=4, label=f"design runs ({best_cat})")
ax.scatter([best["temperature"]], [best["cosolvent"]], s=320, marker="*", color="gold",
           edgecolors="black", linewidths=1.2, zorder=5, label="best feasible")
ax.set_xlabel("temperature (C)")
ax.set_ylabel("cosolvent (%)")
ax.set_title(f"Fitted yield surface, catalyst = {best_cat}", pad=12)
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
save(fig, "wf6_surface.png")

print("\nDONE")
