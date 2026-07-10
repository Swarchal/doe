# End-to-end workflow: a custom design when no recipe fits

Every walkthrough so far reached for a named design. The [first](WORKFLOW.md) and
[screening](WORKFLOW3.md) walkthroughs used a central composite; the
[formulation](WORKFLOW5.md) one used an extreme-vertices simplex. Each is a recipe: tell it
your factors, and it hands back a fixed run set. Recipes are wonderful when your problem is
shaped like the recipe. Real problems often aren't.

This walkthrough is about the escape hatch for when they're not. Three things here each break
the recipes on their own, and they arrive together:

1. one factor is **categorical** — a *choice* (which catalyst?), not a dial — and the
   response-surface generators reject it outright;
2. a **safety constraint** carves an irregular corner out of the design region, so it is no
   longer a box for a recipe to fill;
3. the **run budget is tight and odd** (fifteen runs), matching no factorial or composite count.

The tool for all three is the *coordinate-exchange engine* (`d_optimal`/`i_optimal`): instead of
returning a fixed recipe, it **builds** a run set to support the model you want, over the
budget you have, inside the region you're actually allowed to explore. One pass, one section
per step:

1. write down the factors and watch the recipes bail,
2. express the constraint and build the feasible candidate set,
3. let `d_optimal` choose the fifteen runs,
4. prove the chosen design is good — against a naive one — with design diagnostics,
5. fit the model and find the best setting you're *allowed* to run.

The example: optimizing a catalytic reaction. You must pick one of three catalysts (Pd, Ni, Cu)
and dial in temperature and co-solvent fraction, maximizing percent yield. The catch is a
safety limit — the mixture degrades if run hot *and* co-solvent-rich at once.

> Every console output and figure below is real: it is produced by running these snippets via
> `scripts/build_workflow6_assets.py`, which writes the figures to `docs/img/`. The response is
> simulated from a realistic catalyst-dependent surface so the walkthrough is fully
> reproducible; replace it with your own measurements and the same calls apply.

## 1. Three factors, and why the recipes bail

Two of the knobs are the familiar continuous kind. The third is different: catalyst is a
*categorical* factor — an unordered choice among three discrete options, with no "halfway
between Pd and Ni."

```python
from doe import CategoricalFactor, ContinuousFactor

factors = [
    CategoricalFactor("catalyst", levels=("Pd", "Ni", "Cu")),
    ContinuousFactor("temperature", low=60, high=100, units="C"),
    ContinuousFactor("cosolvent", low=5, high=25, units="%"),
]
```

Reach for a response-surface recipe and it refuses before it starts:

```python
from doe import box_behnken

box_behnken(factors)
```

```text
ValueError: response-surface designs require continuous factors; got non-continuous: ['catalyst']
```

This is not a limitation to work around — it is the recipe being honest. A Box-Behnken or
central composite design places runs at levels like "mid" and "axial," which mean nothing for a
choice among three catalysts. There is no such thing as a catalyst at coded `+1.414`. The
moment one factor is categorical, the whole family of box-filling recipes is off the table, and
you need a method that treats catalyst as three discrete options while still mapping a curved
surface over temperature and co-solvent for each of them.

## 2. The region is not a box

Now the constraint. Physically: the reaction is safe when run hot, and safe when run
co-solvent-rich, but running it hot *and* rich together degrades the product. That rules out one
corner of the temperature/co-solvent square — the design region is no longer a rectangle.

The engine searches over a discrete **candidate set** of allowed points. Start from the full
coded grid and simply drop the points that violate the constraint. In coded units (each factor
scaled to `[-1, +1]`), "not hot *and* rich" is the diagonal line `temperature + cosolvent ≤ 0.5`:

```python
import numpy as np
from doe import candidate_grid

grid = candidate_grid(factors, levels=5)   # catalyst(3) x temperature(5) x cosolvent(5) = 75
# columns are (catalyst, temperature, cosolvent) in coded units
feasible = grid[grid[:, 1] + grid[:, 2] <= 0.5]

print(len(grid), "->", len(feasible))
```

```text
75 -> 57
```

Eighteen of the seventy-five candidates fall in the unsafe corner and are removed; fifty-seven
allowed points remain. That filtered array — nothing more than "the grid, minus the points you
can't run" — *is* how you hand an irregular region to the engine. Any constraint you can write
as a test on the coded coordinates works the same way: build the grid, keep the rows that pass.

![Scatter of the candidate points in the temperature (60-100 C) by co-solvent (5-25%) plane. Blue filled circles mark the feasible candidates filling the lower-left triangle; red crosses mark the excluded points in the upper-right, where a pale red shaded band covers the hot-and-co-solvent-rich corner above the diagonal safety line. The constraint is on the two continuous factors only, so it is identical for every catalyst.](img/wf6_region.png)

The picture is the whole idea: a clean diagonal slices off the unsafe corner, and the engine
will only ever choose from the blue points below it. (The constraint touches only temperature
and co-solvent, so this same feasible triangle applies for each of the three catalysts.)

## 3. Let the engine choose the runs

Now the actual design. Ask for a D-optimal design: fifteen runs, a quadratic model (main
effects, two-factor interactions, and curvature in the continuous factors), searched over the
feasible candidates.

```python
from doe import d_optimal

design = d_optimal(
    factors, n_runs=15, model="quadratic", region=feasible, seed=20260711, n_restarts=40
)
print(f"d_efficiency = {design.meta['d_efficiency']:.3f}, seed = {design.meta['seed']}")
print(design.runs.sort_values(["catalyst", "temperature", "cosolvent"]).round(1).to_string(index=False))
```

```text
d_efficiency = 0.349, seed = 20260711
catalyst  temperature  cosolvent
      Cu         60.0        5.0
      Cu         60.0       15.0
      Cu         60.0       25.0
      Cu         90.0       15.0
      Cu        100.0        5.0
      Ni         60.0        5.0
      Ni         60.0       15.0
      Ni         70.0       25.0
      Ni         80.0        5.0
      Ni        100.0       10.0
      Pd         60.0        5.0
      Pd         60.0       25.0
      Pd         80.0        5.0
      Pd         80.0       20.0
      Pd        100.0        5.0
```

"D-optimal" means the engine picked the fifteen runs that make the model coefficients as
precisely estimable as possible — formally, it maximizes the determinant of the information
matrix `|XᵀX|`, which shrinks the joint uncertainty of the coefficients. It got there by
*coordinate exchange*: seed a random feasible start, then repeatedly swap each run for the
candidate that improves the score most, restarting forty times to dodge local optima. No recipe
was consulted; the run set was constructed.

Read what it chose and the logic shows through. Each catalyst gets exactly five runs — a fair
split of the budget the engine was never told to enforce; it falls out of maximizing
information:

```python
print(design.runs["catalyst"].value_counts().sort_index().to_dict())
```

```text
{'Cu': 5, 'Ni': 5, 'Pd': 5}
```

Within each catalyst, the runs spread to the feasible extremes — the cool/lean corner, the hot
edge, the co-solvent-rich edge — the settings that pin down curvature and interactions. And
every run respects the constraint by construction, because the engine could only ever pick from
the feasible candidates:

```python
coded = design.coded().to_numpy()
print("all runs feasible:", bool(np.all(coded[:, 1] + coded[:, 2] <= 0.5 + 1e-9)))
```

```text
all runs feasible: True
```

## 4. Prove it: diagnostics vs a naive design

"The engine says it's optimal" is not proof. The point of *design diagnostics* is that they
judge any design — however it was made — against a model, so you can compare. Line the D-optimal
design up against the naive thing a hurried practitioner does: grab fifteen allowed points at
random and hope.

```python
from doe import Design, FactorSet, efficiency, condition_number, vif
from doe.analysis.model import build_model_matrix

rng = np.random.default_rng(7)
pick = rng.choice(len(feasible), size=15, replace=False)
# decode_candidates(): coded candidate rows -> a natural-unit runs frame (defined in the
# build script -- factor.decode for the continuous columns, nearest level for the categorical)
naive = Design(decode_candidates(feasible[pick], factors), FactorSet(factors))

def report(name, d):
    eff = efficiency(d, order=2, interactions=True, region=feasible)
    mm = build_model_matrix(d, order=2, interactions=True)
    max_vif = vif(mm.X, term_names=mm.term_names).drop("Intercept", errors="ignore").max()
    print(f"  {name:16s} D={eff.d:.3f}  A={eff.a:.3f}  G={eff.g:.3f}  I={eff.i:.3f}  "
          f"max VIF={max_vif:5.2f}  cond={condition_number(mm.X):5.1f}")

report("naive (random)", naive)
report("D-optimal", design)
```

```text
  naive (random)   D=0.148  A=0.024  G=0.020  I=0.080  max VIF=22.42  cond= 24.0
  D-optimal        D=0.349  A=0.167  G=0.676  I=0.499  max VIF= 5.14  cond=  8.8
```

Every number tells the same story. The **efficiencies** (D/A/G/I, each scaled so an ideal
orthogonal design scores 1.0) are two to twenty-five times higher for the D-optimal design: it
extracts far more information from the same fifteen runs. Its worst **variance-inflation factor**
is 5.1 against the naive design's 22.4 — the naive design has two terms so tangled that their
coefficients are nearly impossible to separate, while the D-optimal design keeps every term
comfortably estimable. And its **condition number** is 8.8 against 24.0, a numerically far
healthier model matrix. Same budget, same region, same model — the difference is entirely in
*which* fifteen points you spend it on.

Where does that leave the alias structure? The correlation matrix among model terms shows it
directly:

![Heatmap of the correlation matrix among the twelve model terms of the D-optimal design (catalyst contrasts, temperature, cosolvent, their interactions, and the two squared terms), colored from blue (-1) through white (0) to red (+1). The diagonal is deep red. Off-diagonals are overwhelmingly pale, near zero, with a few mild patches: the two catalyst contrasts share a light positive correlation and temperature correlates mildly negatively with the temperature-by-cosolvent term.](img/wf6_alias.png)

The off-diagonals are mostly near-white — terms are estimated nearly independently. The few
mild patches are structural, not flaws: the two `catalyst[...]` contrast columns share a
reference level so they correlate a little by construction, and squeezing a design into a
triangular region leaves a slight temperature/interaction coupling the full box wouldn't. This
is about as clean an alias structure as a constrained fifteen-run design can have — and nothing
about it required knowing a recipe.

(The engine also offers `i_optimal`, which minimizes *average prediction variance* over the
region rather than maximizing coefficient precision — the better choice when your goal is
predicting the response across the whole region rather than nailing down individual
coefficients. Same call, `criterion` aside.)

## 5. Fit the model and find the best setting you can run

Run the fifteen and fit the quadratic exactly as any other design — the categorical factor is
expanded automatically into contrast columns, so nothing about the analysis changes:

```python
from doe import fit_ols

measured = design.with_response("yield_pct", yields)   # your fifteen measurements
fit = fit_ols(measured, "yield_pct", model="quadratic")
print(f"R2={fit.r_squared:.3f}  adjR2={fit.adjusted_r2():.3f}  predR2={fit.predicted_r2():.3f}")
print(fit.summary().round(2))
```

```text
R2=0.997  adjR2=0.985  predR2=0.900
                          coefficient  effect  std_error       t     p
term
Intercept                       73.24   73.24       0.63  116.52  0.00
catalyst[Ni]                     3.95    7.90       0.41    9.58  0.00
catalyst[Cu]                    -3.51   -7.01       0.41   -8.48  0.00
temperature                      4.93    9.86       0.69    7.16  0.01
cosolvent                        6.13   12.25       0.69    8.91  0.00
catalyst[Ni]:temperature         3.14    6.28       0.48    6.52  0.01
catalyst[Cu]:temperature        -2.77   -5.54       0.45   -6.10  0.01
catalyst[Ni]:cosolvent           0.19    0.38       0.48    0.39  0.72
catalyst[Cu]:cosolvent           0.41    0.81       0.50    0.81  0.48
temperature:cosolvent            3.17    6.34       0.80    3.96  0.03
temperature^2                   -6.70  -13.40       0.74   -9.09  0.00
cosolvent^2                     -6.20  -12.39       0.74   -8.40  0.00
```

The fifteen runs supported the full twelve-term model with room to spare (R² 0.997, and a
predicted R² of 0.900 that says it generalizes). The story reads straight off the coefficients:
**Ni is the best catalyst** (`catalyst[Ni]` is the largest positive contrast, and its
`catalyst[Ni]:temperature` term says Ni also benefits most from heat, while Cu does worst on
both counts). Yield climbs with both temperature and co-solvent, they reinforce each other
(`temperature:cosolvent` is positive), and the two negative squared terms make a dome. That
combination is the crux: rising, reinforcing, and doming means the surface peaks toward
*high temperature and high co-solvent* — straight into the corner the safety constraint forbids.

So the operating point is a constrained question, and the usual surface optimizer cannot answer
it — it needs all-continuous factors and knows nothing of the constraint:

```python
fit.optimum(maximize=True)
```

```text
TypeError: surface optimization requires all-continuous factors (the coded box); got
non-continuous factor(s) ['catalyst'] ...
```

Answer it the robust way instead — the same move the [formulation walkthrough](WORKFLOW5.md)
makes on the simplex: predict over a dense grid of *feasible* points and take the best. This
naturally handles both the categorical factor (just include each catalyst) and the constraint
(only score allowed points):

```python
# score every catalyst over a fine (temperature, cosolvent) grid, keeping only feasible points
tg, cg = np.meshgrid(np.linspace(60, 100, 81), np.linspace(5, 25, 81))
keep = (tg.ravel() - 80) / 20 + (cg.ravel() - 15) / 10 <= 0.5   # the coded constraint
search = []
for cat in ("Pd", "Ni", "Cu"):
    df = pd.DataFrame({"catalyst": cat,
                       "temperature": tg.ravel()[keep], "cosolvent": cg.ravel()[keep]})
    df["pred"] = np.asarray(fit.predict(df))
    search.append(df)
search = pd.concat(search, ignore_index=True)

best = search.loc[search["pred"].idxmax()]
print(f"best feasible blend: catalyst={best['catalyst']}, "
      f"temperature={best['temperature']:.1f} C, cosolvent={best['cosolvent']:.1f} %")
print(f"predicted yield = {best['pred']:.1f}%")

best_row = pd.DataFrame([{"catalyst": best["catalyst"],
                          "temperature": best["temperature"], "cosolvent": best["cosolvent"]}])
print(fit.predict(best_row, interval="prediction").round(1))
```

```text
best feasible blend: catalyst=Ni, temperature=86.0 C, cosolvent=17.0 %
predicted yield = 80.2%
    fit   se  lower  upper
0  80.2  1.2   76.3   84.1
```

The recommendation: **Ni catalyst at 86 °C and 17% co-solvent, for a predicted 80.2% yield**
(95% prediction interval 76.3–84.1%). And note *where* it lands — exactly on the safety line
`temperature + cosolvent = 0.5` in coded units. The model's true peak lies past it, in the
forbidden corner; the best you can actually run is pressed right up against the constraint. That
is precisely the answer a box-filling recipe could never have given, because it would never have
known the corner was off-limits.

![Filled contour of predicted yield over temperature (60-100 C) and co-solvent (5-25%) for the Ni catalyst. Yield rises from a dim cool/lean corner to a bright yellow ridge running along the diagonal safety line; the hot-and-rich corner above the line is greyed out as forbidden. White circles mark the Ni design runs; a gold star sits on the safety line at 86 C, 17% marking the best feasible setting, just short of the greyed-out peak beyond.](img/wf6_surface.png)

The map makes the constraint's bite visible: the surface would keep climbing into the grey, but
the reachable maximum is the gold star on the edge of it.

**Takeaway.** When your problem doesn't match a recipe — a categorical factor, a constrained or
irregular region, an odd run budget, a custom model — don't distort the problem to fit a design.
Build the design to fit the problem. Filter a candidate grid down to what you're allowed to run,
let `d_optimal` spend your budget where it buys the most information, and *check the result with
design diagnostics* rather than trusting the label. The coordinate-exchange engine is the
general tool the named recipes are convenient special cases of — and it is what
[`augment`](WORKFLOW3.md) quietly used to recycle those screening runs, too.

## To do this, use…

| Step | Function |
| --- | --- |
| Mix a categorical choice with continuous dials | `CategoricalFactor` + `ContinuousFactor` |
| Build the candidate set, then filter to a constrained region | `candidate_grid` + a mask on the coded columns |
| Construct the design for your model, budget, and region | `d_optimal` (or `i_optimal` for prediction variance) |
| Add optimal runs to an existing design | `augment` (see [WORKFLOW3](WORKFLOW3.md)) |
| Judge a design against a model | `efficiency`, `vif`, `condition_number`, `correlation_matrix` |
| Fit a mixed continuous/categorical surface | `fit_ols(..., model="quadratic")` |
| Find the best *feasible* setting | `FitResult.predict` over a dense feasible grid |
