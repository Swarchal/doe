# End-to-end workflow: optimizing a formulation on the simplex

Every walkthrough so far has lived in a *box*. Temperature between 45 and 75 °C, time between 20
and 60 minutes — turn one knob and the others stay put. The [first walkthrough](WORKFLOW.md)
maximized yield in that box; the [two-readout walkthrough](WORKFLOW2.md) balanced two responses
in it; the [screening walkthrough](WORKFLOW3.md) whittled six box factors down to three.

Some of the most common experiments at the bench are not like that at all. When you formulate a
buffer, a growth medium, a solvent system, or — to stay with the transfection theme of
[Vignette 21](VIGNETTES.md) — the lipid mix of a delivery particle, your factors are
**proportions of a whole**. They are not independent: push one component up and the others must
come down, because they always sum to 1. The design region is no longer a box but a *simplex*, a
triangle for three components. Turning one knob turns the others whether you like it or not.

That single constraint changes every step of the pipeline: the design (you cannot just cross
high/low levels — some combinations don't sum to 1), the model (a blend has no meaningful
"intercept", so the fit uses a special no-intercept *Scheffé* form), and the picture (a ternary
contour, not a rectangular one). This walkthrough runs the whole loop for a three-lipid
transfection formulation, one section per step:

1. state the factors honestly as bounded proportions, and see why a box recipe won't build,
2. generate an extreme-vertices design over the constrained simplex,
3. run the blends and attach the measured transfection efficiency,
4. fit a Scheffé blending model — and watch the linear form fail before the quadratic succeeds,
5. read the fitted surface as a ternary map,
6. search the feasible region for the best blend and confirm it.

> Every console output and figure below is real: it is produced by running these snippets via
> `scripts/build_workflow5_assets.py`, which writes the figures to `docs/img/`. The response is
> simulated from a realistic Scheffé surface — poor pure components, a strong
> ionizable-lipid × cholesterol synergy — so the walkthrough is fully reproducible; replace it
> with your own measurements and the same calls apply.

## 1. Factors that sum to one

A mixture factor is a proportion in `[low, high] ⊆ [0, 1]`. Here the three lipids that make up a
transfection particle, each with a formulation-sensible range: you always need a good chunk of
the ionizable lipid, but never all of it; the helper phospholipid and cholesterol fill the rest.

```python
from doe import MixtureFactor

components = [
    MixtureFactor("ion", low=0.20, high=0.80),   # ionizable (cationic) lipid
    MixtureFactor("dope", low=0.05, high=0.50),  # helper phospholipid (DOPE)
    MixtureFactor("chol", low=0.10, high=0.60),  # cholesterol
]
```

For the region to contain any valid blend at all, the bounds have to leave room: the lows must
not already overfill the mixture, and the highs must be able to reach it.

```python
lows = [c.low for c in components]
highs = [c.high for c in components]
print(f"sum of lows  = {sum(lows):.2f}  (must be <= 1)")
print(f"sum of highs = {sum(highs):.2f}  (must be >= 1)")
```

```text
sum of lows  = 0.35  (must be <= 1)
sum of highs = 1.90  (must be >= 1)
```

That leaves a comfortable feasible region. But you cannot design it the way you would a box.
The lattice/centroid recipes for mixtures assume the *full* simplex — pure components allowed —
so with bounds in place they refuse outright:

```python
from doe.generators.mixture import simplex_lattice

simplex_lattice(components, degree=2)
```

```text
ValueError: simplex_lattice requires unconstrained components (low=0, high=1);
constrained component(s) ['ion', 'dope', 'chol'] -- use extreme_vertices instead
```

The error names the fix. When the components are bounded, the design has to respect the polygon
those bounds carve out of the triangle.

## 2. A design on the constrained simplex

`extreme_vertices` builds exactly that: it enumerates the corners of the bounded region
(the McLean–Anderson construction) and adds its centroid — the workhorse design for constrained
formulation problems.

```python
from doe.generators.mixture import extreme_vertices

base = extreme_vertices(components)
print(f"{base.n_runs} runs ({base.meta['n_vertices']} vertices + centroid)")
print(base.runs.round(3).to_string())
print(base.point_types)
```

```text
7 runs (6 vertices + centroid)
     ion   dope   chol
0  0.200  0.200  0.600
1  0.200  0.500  0.300
2  0.350  0.050  0.600
3  0.400  0.500  0.100
4  0.800  0.050  0.150
5  0.800  0.100  0.100
6  0.458  0.233  0.308
('vertex', 'vertex', 'vertex', 'vertex', 'vertex', 'vertex', 'centroid')
```

Every row sums to 1 — that is the constraint, honoured by construction. The bounds have turned
the triangle into a six-sided polygon; the design sits on its corners plus the middle.

![Ternary diagram. The full three-component simplex is drawn as a light-grey triangle with vertices labelled ion (bottom left), dope (bottom right), and chol (top). Inside it, a blue-shaded six-sided polygon marks the feasible region left by the component bounds; its six corners carry red dots (the extreme vertices) and a gold star sits at its centre (the centroid).](img/wf5_region.png)

The picture makes the geometry concrete: the grey triangle is every conceivable three-lipid
blend; the blue polygon is the subset your bounds allow; the red corners and gold centre are the
seven blends the design will actually run. A box design would have tried to sit at the triangle's
own corners — the pure lipids — which are both infeasible here and, as it turns out, terrible
formulations.

Seven runs is the bare skeleton. A quadratic blending model (next section) has six terms, so
seven runs would leave almost nothing to estimate noise with — and no way to check the model.
The honest fix at the bench is replication: run each blend in duplicate. That gives a
model-free read on the run-to-run noise (a pure-error estimate, and with it a lack-of-fit test),
then randomize the run order.

```python
design = base.replicate(2, each=True).randomize(seed=20260710)
print(f"{design.n_runs} runs")
```

```text
14 runs
```

## 3. Run the blends, attach the response

Run the fourteen and record transfection efficiency (percent of cells expressing). The block
below stands in for the bench: it simulates the response from a "true" blending surface in which
pure components perform poorly and the payoff comes from *synergy* — most of all between the
ionizable lipid and cholesterol. Replace it with your own readings.

```python
import numpy as np

props = design.runs[["ion", "dope", "chol"]].to_numpy()
ion, dope, chol = props[:, 0], props[:, 1], props[:, 2]

rng = np.random.default_rng(20260710)
efficiency = (
    20.0 * ion + 15.0 * dope + 20.0 * chol           # each lipid alone: mediocre
    + 30.0 * ion * dope
    + 170.0 * ion * chol                             # the dominant ion x chol synergy
    + 25.0 * dope * chol
    + rng.normal(0.0, 1.2, design.n_runs)
)
measured = design.with_responses(transfection=efficiency)
print(measured.runs.round(3).head(8).to_string())
```

```text
   std_order    ion   dope   chol  transfection
0          0  0.200  0.200  0.600        41.700
1          5  0.350  0.050  0.600        57.438
2          1  0.200  0.200  0.600        43.839
3          9  0.800  0.050  0.150        40.272
4          7  0.400  0.500  0.100        30.144
5          6  0.400  0.500  0.100        30.138
6          2  0.200  0.500  0.300        32.295
7         13  0.458  0.233  0.308        47.008
```

Already a pattern peeks through: the blends heavy in `dope` (rows 4–6, around 30%) trail the
`chol`-rich blends (rows 1–2, above 40%). The two duplicates of the same blend (rows 0 and 2,
both `0.2/0.2/0.6`) land at 41.7 and 43.8 — that gap *is* the noise the lack-of-fit test will
lean on.

## 4. A blending model — linear first, then quadratic

Mixtures are fitted with **Scheffé** polynomials, chosen with the `model="scheffe-..."` names.
They have no intercept: because the proportions always sum to 1, a constant term would be
redundant (it is already hidden in the components). The linear (first-order) blending model says
each component contributes in proportion to how much of it is present — pure additivity, no
interaction:

```python
from doe import fit_ols

fit_lin = fit_ols(measured, "transfection", model="scheffe-linear")
fit_quad = fit_ols(measured, "transfection", model="scheffe-quadratic")

print(f"scheffe-linear    R2 = {fit_lin.r_squared:.3f}   adjR2 = {fit_lin.adjusted_r2():.3f}")
print(f"scheffe-quadratic R2 = {fit_quad.r_squared:.3f}   adjR2 = {fit_quad.adjusted_r2():.3f}")
```

```text
scheffe-linear    R2 = 0.774   adjR2 = 0.733
scheffe-quadratic R2 = 0.987   adjR2 = 0.979
```

The linear model explains 77% of the variation — not nothing, but it is leaving a quarter of the
signal on the table. That gap is the tell: pure additivity is missing something, and in a blend
the thing it misses is *synergy* between components. The quadratic model adds a cross term for
each pair and jumps to R² = 0.987. (These R² values are the ordinary *centered* ones, which are
honest here — the Scheffé columns still span the constant, so there is no inflated-R² trap.)

```python
print(fit_quad.summary().round(2))
```

```text
           coefficient  effect  std_error      t     p
term
ion              19.77     NaN       1.97  10.05  0.00
dope             10.37     NaN       8.27   1.25  0.25
chol             18.99     NaN       5.32   3.57  0.01
ion:dope         34.55     NaN      18.90   1.83  0.10
ion:chol        168.81     NaN      16.42  10.28  0.00
dope:chol        32.31     NaN      19.38   1.67  0.13
```

Read the coefficients as blending effects, not slopes. The linear terms (`ion`, `dope`, `chol`)
are roughly the response you would get from each component alone — all modest, all in the same
ballpark. The story is in the cross terms: `ion:chol` at 168.8 towers over everything, and it is
the one term the data pins down cleanly (t = 10.3). That is the ionizable-lipid × cholesterol
synergy the linear model could not see. (The `effect` column is `NaN` on purpose: an "effect" is
the −1→+1 swing of a coded box factor, which is meaningless for a proportion.)

The mixture ANOVA table tells the same story in the variance budget, following the textbook
convention — the component columns are pooled into a single *Linear blending* row (2 degrees of
freedom for three components), then one row per cross product:

```python
print(fit_quad.anova().round(3))
```

```text
                      SS    df       MS        F      p
Linear blending  736.632   2.0  368.316  241.537  0.000
ion:dope          34.439   1.0   34.439   22.584  0.001
ion:chol         164.508   1.0  164.508  107.882  0.000
dope:chol          4.240   1.0    4.240    2.781  0.134
Residual          12.199   8.0    1.525      NaN    NaN
Total            952.018  13.0      NaN      NaN    NaN
```

The `ion:chol` row (F ≈ 108, p < 0.001) confirms the synergy is real and large; `ion:dope` also
earns its place (p = 0.001), while `dope:chol` does not (p = 0.13). Because we replicated every
blend, we can also ask whether the quadratic model is *adequate* — whether it has missed any
shape the data show — by comparing what it fails to explain against the pure noise between
duplicates:

```python
lof = fit_quad.lack_of_fit()
print(f"lack-of-fit: F = {lof.f_stat:.3f}, p = {lof.p_value:.3f}")
```

```text
lack-of-fit: F = 0.469, p = 0.515
```

A large p-value is the good outcome: the model's residual is no bigger than the noise, so there
is no evidence of a shape it is missing. The quadratic blend is trustworthy.

## 5. Read the surface: the ternary map

`ternary_contour` draws a fitted three-component blend over the simplex, each vertex a pure
component. Pass the design to overlay the blends you actually ran.

```python
from doe.plotting import ternary_contour

ternary_contour(fit_quad, measured, resolution=160)
```

![Ternary contour of the fitted transfection surface over the ion/dope/chol simplex. The surface is bright (high transfection, above 60) along the left edge between the ion and chol vertices, and falls off steeply toward the dope vertex at bottom right, which is deep and dark (below 18). Red dots mark the design blends. Filled colour bands run from about 8 to 64.](img/wf5_ternary.png)

The map reads at a glance. The bright ridge runs along the `ion`–`chol` edge — blends rich in
those two, with the helper lipid low, transfect best. The dark corner is `dope`: pile in the
helper phospholipid and efficiency collapses. This is the `ion:chol` synergy coefficient made
visible — a whole edge of good blends, none of which is a pure component.

## 6. The best feasible blend

The surface suggests where to go, but the recommendation must respect the bounds — and the very
best point on the unconstrained surface may sit outside the feasible polygon. `mixture_candidates`
enumerates a fine grid of blends *inside* the feasible region; score them all through the fit and
take the best.

```python
import pandas as pd
from doe.generators.mixture import mixture_candidates

candidates = mixture_candidates(components, resolution=50)
cand_df = pd.DataFrame({c.name: candidates[:, j] for j, c in enumerate(components)})
pred = np.asarray(fit_quad.predict(cand_df))

best = candidates[pred.argmax()]
print(f"searched {len(candidates)} feasible blends")
print(f"best: ion={best[0]:.2f}, dope={best[1]:.2f}, chol={best[2]:.2f} "
      f"-> {pred.max():.1f}%")
```

```text
searched 478 feasible blends
best: ion=0.48, dope=0.06, chol=0.46 -> 58.0%
```

The recommendation is a near-even split of ionizable lipid and cholesterol with the helper held
down at its floor — exactly where the `ion:chol` ridge said to look. As always the point comes
with a band a confirmation run should land inside:

```python
best_blend = {"ion": 0.48, "dope": 0.06, "chol": 0.46}
print(fit_quad.predict(best_blend, interval="prediction").round(1))
```

```text
    fit   se  lower  upper
0  58.0  1.6   54.3   61.7
```

For contrast, the best *corner* the design actually visited (`ion=0.35, dope=0.05, chol=0.60`)
predicts 55.9% — the interior blend does better, because the optimum lives along an edge the
extreme vertices only bracket. A linear blending model, blind to the synergy, would have had no
way to find it.

![Ternary contour of the same fitted transfection surface, with a large gold star placed on the bright ion–chol ridge at the best feasible blend (ion 0.48, dope 0.06, chol 0.46). The star sits just inside the feasible region, below the surface's unconstrained brightest point at the ion–chol edge.](img/wf5_optimum.png)

The star marks the operating point. Notice it sits *inside* the triangle's edge, not at the
brightest pixel: the unconstrained surface keeps climbing toward the pure ion–chol edge, but the
`dope` floor of 0.05 (and the search over feasible candidates) keeps the recommendation honest.

**Takeaway.** When your factors are proportions that must sum to one, reach for the mixture path,
not the box path. Bounds carve a polygon out of the simplex, so build the design with
`extreme_vertices` (the lattice recipes need the full simplex); fit with a `scheffe-*` blending
model, where the *cross terms are the whole point* — they carry the synergy that a linear blend
cannot represent; read the fit as a `ternary_contour`; and pick the operating point by scoring
`mixture_candidates` so the answer stays inside the feasible region. The same design → fit →
optimize arc as every other walkthrough — on a different-shaped world.

| To do this… | use… |
| --- | --- |
| Declare a component as a bounded proportion | `MixtureFactor(name, low=, high=)` |
| Design over a *bounded* simplex | `extreme_vertices` |
| Design over the *full* simplex | `simplex_lattice`, `simplex_centroid` |
| Fit a blending model | `fit_ols(..., model="scheffe-linear" / "scheffe-quadratic")` |
| Test blending-term significance / adequacy | `FitResult.anova`, `FitResult.lack_of_fit` |
| Map a 3-component surface | `ternary_contour` |
| Optimize inside the feasible region | `mixture_candidates` + `FitResult.predict` |
| Build a D-optimal blend for an odd run budget | `d_optimal(..., region=mixture_candidates(...))` |
