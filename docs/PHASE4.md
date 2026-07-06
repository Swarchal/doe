# Phase 4 — Space-Filling & Mixture Designs

Detailed build plan for Phase 4. Phases 1–3 covered factorial screening, response-surface
methodology, optimization, diagnostics, and computer-generated optimal designs — all for
*independent* factors varied over a box. Phase 4 adds the two remaining generator families
from `PLAN.md`:

1. **Space-filling designs** — Latin Hypercube and Sobol/Halton sequences for computer
   experiments and exploratory sampling, where the goal is uniform coverage of the region
   rather than efficient estimation of a fixed polynomial model.
2. **Mixture designs** — simplex-lattice, simplex-centroid, and extreme-vertices designs for
   formulation problems where the factors are *proportions of a whole* and must sum to 1,
   which changes both the design region (a simplex, not a box) and the model form (Scheffé
   polynomials, no intercept).

The third Phase 4 item in `PLAN.md` — multi-response desirability optimization — **already
shipped in Phase 2b** (`desirability` / `ResponseGoal` in `analysis/optimize.py`), so it is
not re-planned here.

The Phase 1–3 contract is preserved where it applies: generators return a `Design`, analysis
consumes `Design.coded()` → model matrix → OLS. Space-filling fits that contract unchanged.
Mixture designs are the first *deliberate* extension of it: the simplex is not a coded box,
so mixtures introduce a new factor type and a new model-matrix path rather than bending the
`[-1, +1]` coding to fit.

> **Scope decision:** Phase 4 splits cleanly. **Phase 4a** = space-filling
> (`generators/spacefilling.py` + coverage diagnostics) — small, self-contained, everything
> downstream already works. **Phase 4b** = mixture designs (`MixtureFactor`,
> `generators/mixture.py`, Scheffé model support in `analysis/model.py`/`fit.py`, a ternary
> contour plot) — the larger piece, because it touches the modeling layer. 4a and 4b are
> independent; build 4a first for an early, low-risk ship.

> **Scope decision (mixtures):** Phase 4 supports **all-mixture designs only** — every factor
> in the `FactorSet` is a `MixtureFactor`. Mixture-plus-process-variable designs (mixture
> components crossed with ordinary continuous/categorical factors) are a known, well-defined
> extension but multiply the model forms; they are deferred, not designed around.

## Goals

By the end of Phase 4 a user can:

1. Generate a **Latin Hypercube** design (optionally maximin- or correlation-optimized) or a
   **Sobol / Halton** sequence over any set of continuous factors, in natural units, via
   `scipy.stats.qmc` — and judge its coverage with `discrepancy` and `maximin_distance`
   diagnostics.
2. Define **mixture components** (`MixtureFactor`, with optional lower/upper proportion
   bounds) and generate **simplex-lattice**, **simplex-centroid**, and **extreme-vertices**
   designs whose rows sum to 1.
3. Fit **Scheffé mixture polynomials** (linear / quadratic blending models) with the existing
   `fit_ols` front door, get ANOVA / R² / diagnostics on them, and visualize a 3-component
   fitted surface as a **ternary contour plot**.
4. Run a **D-optimal mixture design** for odd run budgets — the extreme-vertices candidate
   set fed straight into Phase 3's `coordinate_exchange`, as promised in `PHASE3.md` §8.

Correctness anchors (continuing the Montgomery/Cornell pattern): LHS stratification (exactly
one point per axis stratum), Sobol' low discrepancy beating i.i.d. sampling, the `{3, 2}`
simplex-lattice's six textbook points, simplex-centroid's `2^k − 1` runs, every mixture row
summing to 1 within tolerance, and a Scheffé quadratic fit reproducing published blending
coefficients from Cornell's *Experiments with Mixtures* yarn-elongation example.

## What's new / what changes

```
src/doe/
  factors.py          # NEW: MixtureFactor (proportion bounds, sum-to-1 contract);
                      #      factor_from_dict dispatch gains "mixture"
  generators/
    spacefilling.py   # NEW: latin_hypercube, sobol, halton (scipy.stats.qmc)
    mixture.py        # NEW: simplex_lattice, simplex_centroid, extreme_vertices,
                      #      mixture_candidates (feeds Phase 3's coordinate_exchange)
  analysis/
    model.py          # CHANGED: build_model_matrix grows a Scheffé path for mixture
                      #          factor sets (no intercept, blending terms)
    fit.py            # CHANGED: fit_ols accepts model="scheffe-linear"/"scheffe-quadratic";
                      #          no-intercept fit path (R², effects semantics documented)
    diagnostics.py    # NEW fns: discrepancy, maximin_distance (coverage metrics)
  plotting.py         # NEW: ternary_contour (3-component mixture surface);
                      #      lazy matplotlib, as today
  serialization.py    # CHANGED: schema accepts the "mixture" factor type
  __init__.py         # exports the new public surface (see §5)
tests/
  test_spacefilling.py  # NEW
  test_mixture.py       # NEW
docs/
  VIGNETTES.md          # two new vignettes (see §7); assets via build_vignette_assets.py
```

New deps: **none required.** `scipy.stats.qmc` (LHS, Sobol', Halton, `discrepancy`) is part
of the existing scipy core dependency. Simplex enumeration and Scheffé expansion are pure
numpy/itertools.

---

## 1. Space-filling designs — `generators/spacefilling.py` (Phase 4a)

Space-filling designs target *coverage*, not model efficiency: for computer experiments,
surrogate modeling, and exploratory screening of expensive simulations there is no assumed
polynomial to be efficient for, so the criterion is "spread the points out."

### 1.1 Generators

```python
def latin_hypercube(
    factors: Sequence[Factor], *, n_runs: int,
    criterion: Literal["maximin", "correlation"] | None = "maximin",
    seed: int | None = None,
) -> Design: ...

def sobol(
    factors: Sequence[Factor], *, n_runs: int,
    scramble: bool = True, seed: int | None = None,
) -> Design: ...

def halton(
    factors: Sequence[Factor], *, n_runs: int,
    scramble: bool = True, seed: int | None = None,
) -> Design: ...
```

- All three require **continuous factors only** (same validation and error message pattern as
  `generators/rsm.py` — a space-filling design over categorical levels is not meaningful).
- Implementation is a thin, well-tested wrapper over `scipy.stats.qmc`:
  `qmc.LatinHypercube` (with `optimization="random-cd"` for `criterion="correlation"` and
  the maximin option for `criterion="maximin"`), `qmc.Sobol`, `qmc.Halton`. Samples come out
  in `[0, 1]^k`, are affinely mapped to coded `[-1, +1]`, then decoded to natural units via
  the existing `ContinuousFactor.decode` — designs are stored in natural units, exactly like
  every other generator.
- `sobol` raises `ValueError` unless `n_runs` is a power of two, with a message naming the
  nearest valid sizes. Sobol' points only achieve their balance guarantees in `2^m` blocks
  (scipy itself warns otherwise); making it an error keeps the generated designs defensible,
  and `halton`/`latin_hypercube` are the escape hatch for arbitrary `n_runs`.
- Reproducibility and self-description via `meta`, matching Phase 3's convention:
  `meta["sampler"]` (`"lhs"`/`"sobol"`/`"halton"`), `meta["seed"]`, `meta["criterion"]` /
  `meta["scramble"]`. `point_types` is left `None` — there are no special runs.

### 1.2 Coverage diagnostics — `analysis/diagnostics.py`

Model-based diagnostics (`efficiency`, `vif`, …) already work on these designs unchanged; what
is missing is the pair of *model-free* coverage metrics space-filling is judged by:

```python
def discrepancy(design: Design, *, method: str = "CD") -> float: ...      # qmc.discrepancy
def maximin_distance(design: Design) -> float: ...  # min pairwise distance, coded units
```

Both operate on `Design.coded()` rescaled to the `[0, 1]^k` unit cube (the convention
`qmc.discrepancy` expects), so they judge *any* design, not just Phase 4a's — e.g. comparing
a CCD's coverage against an LHS with the same budget. Lower discrepancy = more uniform;
larger maximin distance = better separated. Pure numpy/scipy, headless, individually
testable — the established `diagnostics.py` pattern.

---

## 2. Mixture designs — Phase 4b

Mixture problems change three things at once, and the plan keeps them separable:
the **factor type** (§2.1), the **design generators** (§2.2), and the **model form** (§2.3).

### 2.1 `MixtureFactor` — `factors.py`

```python
@dataclass(frozen=True)
class MixtureFactor:
    name: str
    low: float = 0.0     # lower bound on the proportion
    high: float = 1.0    # upper bound on the proportion
    units: str | None = None
```

- Proportions are already dimensionless and bounded, so mixture columns are **not** rescaled
  to `[-1, +1]`: `Design.coded()` passes mixture columns through as proportions. This is a
  documented, deliberate exception to the box-coding contract — the simplex has no center/
  half-range coding that preserves the sum-to-1 constraint, and Scheffé models are defined on
  proportions. (Pseudo-component transformation for tightly constrained regions is a deferred
  refinement; see §8.)
- `FactorSet` gains a `is_mixture` property (`True` iff *all* factors are `MixtureFactor`)
  and validation: mixing `MixtureFactor` with other factor types raises (per the scope
  decision above), `Σ low ≤ 1 ≤ Σ high` must hold or the simplex region is empty.
- `to_dict`/`from_dict` with `"type": "mixture"`; `factor_from_dict` and the serialization
  schema (`serialization.py`, `docs/SERIALIZATION.md`) gain the new tag.

### 2.2 Generators — `generators/mixture.py`

```python
def simplex_lattice(factors, *, degree: int) -> Design: ...     # {k, m} lattice
def simplex_centroid(factors) -> Design: ...                     # 2^k − 1 runs
def extreme_vertices(factors, *, n_centroids: int = ...) -> Design: ...
def mixture_candidates(factors, *, resolution: int = 10) -> np.ndarray: ...
```

- **`simplex_lattice`** — all compositions with proportions in `{0, 1/m, …, 1}` summing to 1
  (`itertools` composition enumeration; `C(k+m−1, m)` runs). Requires unconstrained
  components (`low == 0, high == 1`); constrained problems go to `extreme_vertices`.
- **`simplex_centroid`** — the `2^k − 1` centroids of every non-empty subset of components
  (pure blends, binary 50/50s, …, overall centroid).
- **`extreme_vertices`** — for constrained regions (`low`/`high` bounds active): enumerate
  the vertices of the constrained simplex (XVERT-style: all bound combinations on `k−1`
  components, remaining component takes up the slack, keep feasible points), optionally
  append edge/face/overall centroids. This is the constrained-formulation workhorse.
- **`mixture_candidates`** — a discrete candidate set over the constrained simplex (lattice
  points filtered to the feasible region, plus the extreme vertices and centroids). Shaped
  exactly like Phase 3's `candidate_grid` output so it feeds `coordinate_exchange` /
  `d_optimal(..., region=...)` directly — D-optimal mixture designs for odd run budgets come
  free from the Phase 3 engine, which is the payoff `PHASE3.md` §8 promised. (The engine's
  row exchanges swap whole candidate points, so the sum-to-1 constraint is preserved by
  construction — every candidate already satisfies it.)
- `point_types` tags rows (`"vertex"`, `"edge-centroid"`, `"centroid"`, …) so replicated
  centroids drive lack-of-fit exactly as center points do today. `meta` records the recipe
  (`degree`, bounds, etc.).

### 2.3 Scheffé models — `analysis/model.py` + `analysis/fit.py`

Because the proportions sum to 1, the intercept is confounded with the sum of the linear
terms — mixture models drop the intercept and the pure-quadratic terms:

- **Scheffé linear:** `ŷ = Σ βᵢ xᵢ`
- **Scheffé quadratic:** `ŷ = Σ βᵢ xᵢ + ΣΣ_{i<j} βᵢⱼ xᵢ xⱼ`

Plan:

- `build_model_matrix` detects `factors.is_mixture` and emits the Scheffé expansion
  (no intercept column; `order=1` → linear blending, `order=2` → adds the `i<j` cross
  products; `interactions` is ignored/validated for mixtures). Term names are the component
  names and `A:B` products — consistent with existing naming, so `anova_table`, `vif`,
  `correlation_matrix`, and the plots that key off `term_names` work unchanged.
- `fit_ols` grows the model literals `"scheffe-linear"` / `"scheffe-quadratic"` (the
  `ModelSpec` pattern) and a **no-intercept fit path**: R² is computed against the *uncorrected*
  or centered total consistently (decide once, test it, document it — statsmodels' no-intercept
  R² gotcha is the trap to avoid); `effects` (the −1→+1 swing, `2×coef`) is meaningless for
  blending coefficients, so `FitResult.effects` is `NaN`/omitted for mixture fits and the
  docstring says why. `anova_table`'s sequential SS already comes from the model matrix via QR,
  so it needs only the no-intercept total-SS convention, not a rework.
- `stationary_point` / `optimum` are **not** extended to the simplex in Phase 4 —
  `desirability` and `optimum` assume the coded box. Constrained optimization over the
  simplex is deferred (§8); the vignette shows reading the optimum off the ternary contour.

### 2.4 Ternary plot — `plotting.py`

```python
def ternary_contour(result: FitResult, design: Design, *, resolution: int = 100, ax=None): ...
```

For exactly 3 components: barycentric → Cartesian transform, predict over a triangular grid
via the Scheffé model matrix, `tricontourf` + labeled component axes, design points overlaid.
Headless core (`ternary_grid`) separated from the matplotlib wrapper, mirroring
`surface_grid`/`contour_plot`. >3 components raises with a message suggesting fixing
components (a `fixed=` slice, like `contour_plot`, is a possible follow-up, not Phase 4 scope).

---

## 3. Public API additions (`__init__.py`)

Generators: `latin_hypercube`, `sobol`, `halton`, `simplex_lattice`, `simplex_centroid`,
`extreme_vertices`, `mixture_candidates`. Factors: `MixtureFactor`. Diagnostics:
`discrepancy`, `maximin_distance`. Plotting stays in `doe.plotting` (`ternary_contour`).
Flat top-level namespace, as established.

---

## 4. Tests (anchors)

**`test_spacefilling.py`**
- LHS stratification: for every factor, each of the `n_runs` equal coded strata contains
  exactly one point.
- `latin_hypercube(..., seed=0)` twice is bit-identical; `meta` records sampler/seed.
- `sobol` with `n_runs=8` succeeds, `n_runs=10` raises naming 8 and 16; scrambled Sobol' has
  lower `discrepancy` than an i.i.d.-uniform design of the same size (seeded).
- `discrepancy`/`maximin_distance` sanity: the `2^2` full factorial's maximin distance in the
  unit cube is exactly 1.0 (side length); adding a duplicate run drives maximin to 0.
- Categorical factor → `ValueError` for all three generators.

**`test_mixture.py`**
- `{3, 2}` simplex lattice = the 6 textbook points (3 pure + 3 binary); `{3, 3}` has 10 runs.
- `simplex_centroid` on `k=3` has 7 runs; every generated design's rows sum to 1 within 1e-12.
- `extreme_vertices` with Cornell/McLean-Anderson published bounds reproduces the published
  vertex set; infeasible bounds (`Σ low > 1`) raise.
- Scheffé quadratic on Cornell's yarn-elongation data recovers the published blending
  coefficients to tolerance; the model matrix has no intercept column and `C(3,2)+3 = 6` terms.
- `d_optimal(factors, n_runs=7, region=mixture_candidates(...))` returns rows that all sum
  to 1 and scores `log|XᵀX|` ≥ the simplex-centroid design's for the same model.
- `MixtureFactor` round-trips through `to_dict`/`factor_from_dict` and design serialization.
- Mixing `MixtureFactor` with `ContinuousFactor` in one `FactorSet` raises.

Run gates unchanged: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.

---

## 5. Build order

**Phase 4a (independent, ship first):**
1. `generators/spacefilling.py`: `latin_hypercube`, `sobol`, `halton` + `test_spacefilling.py`.
2. `diagnostics.discrepancy` / `maximin_distance` + tests.
3. Vignette: "Space-filling designs for computer experiments" (generate LHS vs Sobol', compare
   `discrepancy`, fit and plot as usual) — re-run `scripts/build_vignette_assets.py`.

**Phase 4b (factor type → generators → model → plot, each step green):**
4. `MixtureFactor` + `FactorSet.is_mixture` + serialization dispatch + tests.
5. `simplex_lattice` / `simplex_centroid` / `extreme_vertices` / `mixture_candidates`
   (design generation only; analysis not yet wired) + lattice/centroid/vertex anchors.
6. Scheffé path in `build_model_matrix`, no-intercept path in `fit_ols` (+ `anova_table`
   total-SS convention) + the Cornell coefficient anchor.
7. `ternary_contour` (headless `ternary_grid` core first) + plot smoke tests.
8. D-optimal-mixture integration test (Phase 3 engine over `mixture_candidates`).
9. Vignette: "Mixture designs" (constrained formulation → extreme vertices → Scheffé fit →
   ternary contour) — re-run `scripts/build_vignette_assets.py`; update `PLAN.md`'s roadmap
   line and `CLAUDE.md`'s status paragraph.

Test-first throughout; each step keeps pytest + ruff + mypy green.

---

## 6. Resolved decisions and deferred work

- **Desirability:** already shipped in Phase 2b; Phase 4 does not touch it.
- **All-mixture only:** mixture-process-variable (combined) designs are deferred — they
  multiply model forms (Scheffé × process crossings) for a niche payoff at this stage.
- **No simplex coding:** mixture columns stay as proportions through `coded()`;
  pseudo-component transformation for tight constrained regions is a deferred refinement.
- **Sobol' run counts:** hard error off powers of two rather than a warning — defensible
  designs over convenience; Halton/LHS cover arbitrary `n`.
- **Simplex optimization:** `optimum`/`desirability` over the simplex (sum-to-1 constrained
  search) is deferred; the ternary contour is the Phase 4 answer for locating optima.
- **Ternary for >3 components:** deferred (`fixed=`-style slicing would mirror `contour_plot`).

---

## 7. After Phase 4 (Phase 5 preview)

With Phase 4 done, every generator family in `PLAN.md` §2 is covered except
**classical/blocking**: randomized complete block designs, Latin squares, blocked factorials
(assigning fractions to blocks via defining contrasts), and richer run-order utilities.
That — plus the deferred items above (combined mixture-process designs, simplex-constrained
optimization, pseudo-components) — is the natural Phase 5 pool.
