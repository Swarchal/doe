# Phase 3 — Optimal Designs & Design Diagnostics

Detailed build plan for Phase 3. Phases 1–2 gave us *named* designs (factorial,
Plackett–Burman, CCD, Box–Behnken) — recipes that produce a fixed run set for a fixed
model. Phase 3 adds the two pieces that make a DoE library general-purpose:

1. **Design diagnostics** — evaluate *any* design (however it was made) against a model:
   condition number, VIF, leverage, the alias/correlation matrix, and D-/A-/G-/I-efficiency.
2. **Optimal (computer-generated) designs** — a coordinate-exchange engine that *builds* a
   run set to maximize a chosen criterion (D- or I-optimality) for a user-specified model,
   run budget, and candidate region. This is what handles irregular constraints, custom
   models, odd run counts, and augmenting an existing design — the cases the named recipes
   can't.

The Phase 1–2 contract is unchanged: everything still flows through a coded design matrix
(`Design.coded()` → `build_model_matrix` → analysis). Diagnostics read that matrix;
the optimal-design engine *searches over* it. Phase 3 extends each link, never replaces it.

> **Scope decision:** Phase 3 splits cleanly. **Phase 3a** = design diagnostics (a new
> `analysis/diagnostics.py`, consolidating the alias matrix that already lives in
> `plotting.py`). **Phase 3b** = the coordinate-exchange optimal-design engine
> (`generators/optimal.py`) with D- and I-optimality, plus design *augmentation*. 3a is a
> prerequisite for 3b — the exchange engine's objective *is* a diagnostic (the D-criterion),
> so building and testing diagnostics first gives the engine a verified objective to climb.

> **Implementation status:** Phase 3 is implemented. Diagnostics, alias/leverage plotting,
> candidate grids, mixed continuous/categorical candidate regions, D- and I-optimal design
> wrappers, coordinate exchange, and augmentation are all covered by tests. The exchange engine
> currently uses a correctness-first discrete candidate exchange with full model-matrix and
> objective recomputation on each trial; rank-1 determinant updates and continuous line search are
> deferred optimizations, not missing user-facing Phase 3 features.

## Goals

By the end of Phase 3 a user can:

1. Take any `Design` + a model spec and get a **diagnostics report**: `|XᵀX|`, condition
   number, per-term VIF, leverage (hat diagonal) per run, and D/A/G/I-efficiency relative to
   an orthogonal reference — to *judge* a design before running it.
2. Render the **alias/correlation matrix** as a heatmap (already shipped in `plotting.py`;
   3a re-homes its headless core in `diagnostics.py` and leaves the plot a thin wrapper).
3. **Generate a D-optimal or I-optimal design** for an arbitrary model and run budget over a
   discrete candidate set, including grids over continuous factors and categorical levels, via
   coordinate exchange with random restarts.
4. **Augment** an existing design with `n` extra optimal runs (same engine, seeded with the
   fixed rows) — the common "I already ran 8, give me 4 more that help most" workflow.

Correctness anchors (continuing the Montgomery pattern): a full `2^k` factorial is D-optimal
for the first-order-plus-interaction model and scores efficiency `1.0`; a saturated design has
leverages summing to `p` with at least one run at `h=1`; a generated D-optimal design matches
the determinant of a known textbook optimal design to tolerance; an orthogonal design has VIF
≈ 1 for every term.

## What's new / what changes

```
src/doe/
  analysis/
    diagnostics.py    # DONE: condition number, VIF, leverage, alias/correlation matrix,
                      #      D/A/G/I-efficiency. Headless, numpy-only. The alias-matrix core
                      #      moves here from plotting.py.
    model.py          # DONE: add expand_coded_points -- the array-based companion to
                      #       build_model_matrix, shared by I-optimality and G/I-efficiency.
                      #       Mixed continuous/categorical candidate rows are supported.
  generators/
    optimal.py        # DONE: candidate_grid, coordinate_exchange, d_optimal, i_optimal,
                      #       augment. (The PLAN.md slot is filled.)
  plotting.py         # DONE: alias_matrix is a thin wrapper over diagnostics; leverage_plot
                      #       added. (lazy matplotlib, as today)
  __init__.py         # DONE: exports the new public surface (see §5)
tests/
  test_diagnostics.py # NEW
  test_optimal.py     # NEW
```

New deps: **none required.** Everything is `numpy.linalg` (determinants, `slogdet` for stable
log-determinants, `lstsq`/`pinv`, eigenvalues) on the existing model matrix. `scipy.stats.qmc`
(already available via the scipy core dep) seeds random restarts. No `statsmodels`.

---

## 1. Design diagnostics — `analysis/diagnostics.py` (Phase 3a)

All functions take a model matrix `X` (or a `Design` + model spec they expand via
`build_model_matrix`, mirroring how `anova_table` consumes a design). They are pure numpy,
headless, and individually testable — the same split that made `surface_grid` the tested core
under `contour_plot` in Phase 2.

### 1.1 Information-matrix scalars

```python
def information_matrix(X: np.ndarray) -> np.ndarray: ...        # XᵀX
def condition_number(X: np.ndarray) -> float: ...              # σ_max/σ_min of X (SVD)
def log_det_information(X: np.ndarray) -> float: ...           # slogdet(XᵀX), the D-objective
```

`slogdet` (not `det`) so the determinant of a large model matrix doesn't overflow/underflow —
this is also exactly what the exchange engine in §2 maximizes, so it lives here and is shared.

### 1.2 Variance inflation & leverage

```python
def vif(X: np.ndarray, *, term_names: Sequence[str] | None = None) -> pd.Series: ...
def leverage(X: np.ndarray) -> np.ndarray: ...   # diag(H), H = X(XᵀX)⁻¹Xᵀ
```

- **VIF** per non-intercept term = `1 / (1 − R²ⱼ)` where term *j* is regressed on the others.
  Computed from the diagonal of the *correlation*-scaled `(XᵀX)⁻¹` (standard closed form) so we
  don't run `p` separate regressions. VIF ≈ 1 ⇒ orthogonal; >5–10 ⇒ troublesome collinearity.
- **Leverage** = hat-matrix diagonal; reuse the `H` computation already used for PRESS in
  `anova.py` (factor it out rather than duplicate). `Σ hᵢ = p` (number of terms) is the test
  invariant. Feeds a new `leverage_plot` diagnostic in `plotting.py`.

### 1.3 Alias / correlation matrix (re-home, don't rebuild)

`plotting.py` already has `alias_matrix` (pairwise term correlations = the design's alias
structure). Phase 3a extracts its headless core to `diagnostics.py`:

```python
def correlation_matrix(X: np.ndarray, term_names: Sequence[str]) -> pd.DataFrame: ...
```

`plotting.alias_matrix` becomes a thin heatmap wrapper over it — the same `surface_grid` /
`contour_plot` pattern. No behaviour change; existing `test_plotting.py` alias tests stay green.

### 1.4 Optimality efficiencies

```python
@dataclass(frozen=True)
class Efficiency:
    d: float   # (|XᵀX|^(1/p) / n)               relative to an orthogonal design = 1
    a: float   # p / trace((XᵀX)⁻¹), normalized   A-efficiency
    g: float   # based on max prediction variance over the region
    i: float   # average prediction variance over the region (I/V-optimality)

def efficiency(design, *, order=1, interactions=True, region=None) -> Efficiency: ...
```

**Model-spec convention.** `efficiency` selects its model with `order`/`interactions`, exactly
like `build_model_matrix` and `anova_table` — the *diagnostics* convention. The design-*building*
generators in §2 take a `model="linear"|"quadratic"` string instead, for ergonomics (like
`fit_ols`). Defaulting `order=1` matches `build_model_matrix`; pass `order=2` to judge a
response-surface design.

- **D-efficiency** `(|XᵀX|^{1/p} / n)` — normalized so an orthogonal (e.g. full-factorial)
  design scores `1.0`. The headline number.
- **A-efficiency** from `trace((XᵀX)⁻¹)` (average coefficient variance).
- **G-efficiency** from the *maximum* scaled prediction variance over the candidate region;
  **I-efficiency** from the *average* (integrated) scaled prediction variance. Both need a
  region to integrate/maximize over — default to a Monte-Carlo / grid sample of the coded box
  (reuse the candidate-set machinery from §2 so the region definition is shared).

G and I are the place to keep scope honest: implement them over a **sampled** candidate region
(grid or QMC), not closed-form integration. That's standard, good enough, and consistent with
how the exchange engine evaluates I-optimality.

**Shared core — `expand_coded_points`.** Evaluating the scaled prediction variance
`f(x)ᵀ (XᵀX)⁻¹ f(x)` at region points needs to expand an arbitrary `(m, k)` array of *coded
points* into model rows `f(x)` — but `build_model_matrix` only expands a `Design`. So Phase 3
adds `model.expand_coded_points(points, factors, *, order, interactions)`: the array-based
companion with identical `term_names`. It now handles mixed continuous/categorical factors too:
continuous columns are coded values in `[-1, +1]`, while categorical columns use discrete coded
coordinates from `np.linspace(-1, 1, n_levels)` and are decoded back to natural labels before
effect coding. Both the G/I region sampler here and the I-optimal engine in §2 consume it, so it
lives in `model.py` and is written once.

---

## 2. Optimal designs — `generators/optimal.py` (Phase 3b)

The engine that *builds* a design. Coordinate exchange (Meyer–Nachtsheim) is the workhorse:
cheaper than Fedorov point-exchange, handles continuous factors, and parallelizes over restarts.

### 2.1 Candidate region

```python
def candidate_grid(factors, *, levels=3) -> np.ndarray: ...   # discrete grid in coded units
```

Implemented region mode, in coded units:

- **Discrete candidate set** — an explicit array of allowed coded points (grid, or
  user-supplied to encode constraints / categorical levels / disallowed combinations). This is
  what makes constraints expressible: just omit infeasible points from the set.

`candidate_grid` builds the default finite set. Continuous factors take evenly spaced values in
`[-1, +1]`; categorical factors always take their discrete level coordinates. The region object
is shared with §1.4's G/I efficiency sampling. A true continuous-box line search is a possible
future optimization, but Phase 3 currently searches the supplied discrete candidate set.

### 2.2 Coordinate-exchange engine

```python
@dataclass(frozen=True)
class OptimalDesign:
    design: Design
    criterion: str            # "D" | "I"
    score: float              # log|XᵀX| (D) or avg prediction variance (I)
    d_efficiency: float
    n_restarts: int
    converged: bool

def coordinate_exchange(
    factors, *, n_runs, model="quadratic", criterion="D",
    region=None, n_restarts=20, seed=None, fixed_runs=None, max_iter=100,
) -> OptimalDesign: ...
```

Algorithm:

1. **Seed** an `n_runs × k` start (random feasible points; for *augmentation*, the first
   `len(fixed_runs)` rows are `fixed_runs` and are never exchanged).
2. **Sweep** every mutable run; replace the whole row with the candidate point that most improves
   the criterion. Each trial recomputes the full model matrix and objective from scratch. This is
   slower than a determinant-update implementation but deliberately simple and correctness-first.
3. **Iterate** sweeps until no coordinate improves (or `max_iter`).
4. **Restart** `n_restarts` times from fresh random seeds; keep the best. Guards against the
   local optima coordinate exchange is prone to.

`criterion="D"` maximizes `log|XᵀX|` (via §1.1's `log_det_information`); `criterion="I"`
minimizes average prediction variance over the region. Both criteria work for mixed
continuous/categorical candidate regions.

### 2.3 Public generators

Thin, intention-revealing wrappers over the engine (matching the named-design ergonomics):

```python
def d_optimal(factors, *, n_runs, model="quadratic", **kw) -> Design: ...
def i_optimal(factors, *, n_runs, model="quadratic", **kw) -> Design: ...
def augment(design, *, n_runs, model="quadratic", criterion="D", **kw) -> Design: ...
```

`**kw` forwards the engine options (`region`, `n_restarts`, `seed`, `max_iter`) — so
`d_optimal(factors, n_runs=12, seed=0)` is reproducible through the thin wrapper; `d_optimal`/
`i_optimal` fix `criterion`, and `augment` additionally passes `fixed_runs` = the existing
design's coded rows.

They return a plain `Design` (so the rest of the library — `coded()`, fitting, ANOVA, plots —
works unchanged), with the `OptimalDesign` diagnostics carried in `design.meta` (criterion,
score, d_efficiency, n_restarts, seed) so the result is reproducible and self-describing, the
same way CCD stashes `alpha` / `axial_extrapolates` today. `augment` tags the original rows
`point_type="existing"` and the new rows `"augment"`.

---

## 3. Plotting — `plotting.py`

- `alias_matrix` → thin wrapper over `diagnostics.correlation_matrix` (re-home, §1.3).
- **NEW** `leverage_plot(result_or_design, ax=None)` — leverage per run with the `2p/n`
  high-leverage reference line; the design-evaluation companion to the residual diagnostics.

Lazy matplotlib, `[plotting]` extra only — unchanged convention.

---

## 4. Public API additions (`__init__.py`)

Generators: `d_optimal`, `i_optimal`, `augment`, `coordinate_exchange`, `candidate_grid`
(+ the `OptimalDesign` dataclass). Diagnostics: `efficiency`, `vif`, `leverage`,
`condition_number`, `correlation_matrix`, `information_matrix`, `log_det_information`
(+ the `Efficiency` dataclass). Keep the flat top-level namespace. `diagnostics.*` headless
cores are importable from `doe.analysis.diagnostics`; the plotting wrappers stay in
`doe.plotting` (the Phase 1–2 convention). The shared `expand_coded_points` core stays internal
to `doe.analysis.model` (not re-exported — it's machinery, like `build_model_matrix`).

---

## 5. Tests (anchors)

**`test_diagnostics.py`**
- A `2^3` full factorial fit with the first-order+interaction model is **orthogonal**:
  every VIF == 1, condition number == 1, D-efficiency == 1.
- `Σ leverage == p` for any design; a saturated design has a run with `h == 1`.
- Correlation matrix of an orthogonal design is the identity; a Plackett–Burman main effect
  shows the known `±1/3` partial aliasing with a two-factor interaction (anchors the existing
  alias behaviour after the re-home).
- D/A/G/I efficiencies of a known textbook design match published values to tolerance.

**`test_optimal.py`**
- For the first-order+interaction model with `n_runs == 2^k`, `d_optimal` recovers (a
  relabeling of) the **full factorial** — `meta["d_efficiency"] == 1`, `|XᵀX|` equals the
  factorial's.
- A D-optimal quadratic design on `k=2`, `n_runs=9` reproduces the known optimal `|XᵀX|`
  (the `3²` grid) to tolerance, and `d_optimal(..., seed=0)` twice is bit-identical (the seed
  forwarded through the wrapper to `coordinate_exchange`).
- `augment` keeps the existing rows byte-for-byte at the front and only adds `n_runs` new rows
  (tagged `point_type` `"existing"`/`"augment"`), and the augmented `|XᵀX|` ≥ the original
  (extra optimal runs never reduce information).
- I-optimal design has **lower average prediction variance** than the D-optimal one for the
  same model/budget — compared via `efficiency(...).i` evaluated on a **shared region**, since
  the two designs' stored `meta["score"]` are in different units (a log-det vs. an average
  variance) and aren't directly comparable. The reported D score agrees with a full `slogdet`
  refit of the final design to numerical tolerance.

Run gates unchanged: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.

---

## 6. Build order

**Phase 3a (diagnostics first — the engine's objective must be verified before it's climbed):**
1. DONE: `analysis/diagnostics.py`: `information_matrix`, `condition_number`, `log_det_information`,
   `leverage` (factored out of `anova.py`'s PRESS hat matrix), `vif` + `test_diagnostics.py`.
2. DONE: Re-home the alias core: `correlation_matrix` in `diagnostics.py`, `plotting.alias_matrix`
   becomes a wrapper; add `leverage_plot`. Keep `test_plotting.py` green.
3. DONE: `model.expand_coded_points` (array-based companion to `build_model_matrix`), then
   `efficiency` (D/A/G/I) over a sampled region on top of it + tests.

**Phase 3b (the engine, on top of verified diagnostics):**
4. DONE: Candidate region support, including mixed continuous/categorical candidate coordinates,
   reusing `expand_coded_points`.
5. DONE: `coordinate_exchange` with random restarts and full objective recomputation; verify the
   reported D score against a full refit in tests.
6. DONE: `d_optimal` / `i_optimal` / `augment` wrappers + `test_optimal.py`; stash diagnostics in
   `design.meta`.

Test-first throughout, matching Phase 2's cadence. Each step keeps the full suite + ruff +
mypy green.

---

## 7. Resolved decisions and deferred optimizations

- **Continuous coordinate search:** Phase 3 uses a discrete candidate grid first: simpler,
  deterministic, constraint-friendly, and compatible with categorical factors. True 1-D line
  search can be revisited if precision near continuous optima becomes important.
- **G/I region:** efficiencies and I-optimal search use sampled/discrete candidate regions, not
  closed-form integration.
- **Engine choice:** Phase 3 implements coordinate exchange only. Fedorov point-exchange remains a
  possible later addition for pure candidate-set problems.
- **Rank-1 determinant updates:** deferred. The current engine recomputes the full model matrix
  and `slogdet` for correctness; determinant-update shortcuts can be added behind the same tests.
- **`meta` vs. a returned diagnostics object:** generators return a plain `Design` (with
  diagnostics in `meta`) so downstream code is uniform; `coordinate_exchange` additionally
  returns the richer `OptimalDesign` for callers who want the full search report. (Mirrors
  Phase 2b's `Optimum` / `StationaryPoint` dataclasses.)

---

## 8. After Phase 3 (Phase 4 preview)

With optimal designs and diagnostics done, the roadmap's remaining specialized generators
(PLAN.md Phase 4) are: **space-filling** (Latin Hypercube + Sobol/Halton via `scipy.stats.qmc`)
for computer experiments, and **mixture designs** (simplex-lattice / simplex-centroid /
extreme-vertices) for constrained-proportion problems. Both reuse Phase 3's candidate-region
and diagnostics machinery — space-filling is a region-sampling problem, and extreme-vertices
mixture designs are a constrained candidate set the coordinate-exchange engine can already
consume.
