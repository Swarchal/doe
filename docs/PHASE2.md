# Phase 2 — Response Surface Methodology (RSM)

Detailed build plan for Phase 2. Phase 1 gave us screening (factorial designs + linear/
interaction OLS). Phase 2 adds the machinery to **fit curvature** with second-order designs
(CCD, Box–Behnken) and proper inference (ANOVA, lack-of-fit, confidence intervals).

The Phase 1 contract is unchanged: everything still flows through a coded design matrix
(`Design.coded()` → `build_model_matrix` → `fit_ols`). Phase 2 extends each link rather than
replacing it.

> **Scope decision:** Phase 2 is split. **Phase 2a** = designs + quadratic fitting +
> ANOVA/lack-of-fit + contour plots. **Phase 2b** = surface optimization (stationary
> point, canonical analysis, constrained optimum), multi-response desirability, and 3-D surface
> plots (detailed in §8). Both phases are now implemented.

> **Status (2026-06-23):** Phase 2a **and** Phase 2b are **complete**. Phase 2a: generators
> (`rsm.py`), center-point tracking on `Design`, the quadratic model cleanup, `FitResult`
> inference fields, `analysis/anova.py`, and the §4 plotting helpers (contour + residual/
> half-normal diagnostics). Phase 2b: `analysis/optimize.py` (`stationary_point` + canonical
> analysis, constrained `optimum`, Derringer-Suich `desirability`) and the 3-D `surface_plot`.
> 76 tests pass with ruff + mypy clean. Progress is tracked per item with ✅ below.

## Goals (Phase 2a)

By the end of Phase 2a a user can:

1. Generate a central composite or Box–Behnken design for *k* continuous factors, with the
   right axial distance and a chosen number of center points.
2. Fit the full second-order model `ŷ = b₀ + Σbᵢxᵢ + Σbᵢⱼxᵢxⱼ + Σbᵢᵢxᵢ²`.
3. Get an ANOVA table with per-term sums of squares, a **lack-of-fit test** against pure error
   (from replicated center points), and R² / adjusted R² / predicted R² (Q²).
4. Draw contour plots of the fitted surface, plus residual diagnostics.

Locating the optimum / canonical analysis / desirability is **Phase 2b** (§8).

Correctness anchor (continues the Phase 1 pattern): reproduce canonical designs and worked
examples from Montgomery, *Design and Analysis of Experiments* (the RSM chapters), in tests.

## What's new / what changes (Phase 2a)

```
src/doe/
  generators/
    rsm.py          # ✅ DONE: central_composite, box_behnken
  analysis/
    model.py        # ✅ DONE: drops x² for pure ±1 factors (keeps full rank)
    fit.py          # ✅ DONE: model="quadratic"; FitResult + cov/SE/CI/t/p + dof
    anova.py        # ✅ DONE: anova_table, lack_of_fit (LackOfFit), PRESS / pred. R²
  design.py         # ✅ DONE: center-point tracking (point_types, n_center, center_indices)
  plotting.py       # ✅ DONE: surface_grid, contour_plot, residuals_vs_fitted, normal_qq,
                    #          half_normal_plot (lazy matplotlib; fit.py gains FitResult.factors)
tests/
  test_rsm.py       # ✅ DONE
  test_anova.py     # ✅ DONE
  test_model.py     # ✅ DONE: + quadratic square-dropping tests
  test_plotting.py  # ✅ DONE: surface_grid numerics + Axes/labels for each plot
  conftest.py       # ✅ DONE: forces the headless Agg matplotlib backend
```

New top-level exports added: `central_composite`, `box_behnken`, `anova_table`, `lack_of_fit`,
`LackOfFit`, `press`, `predicted_r2`, `adjusted_r2`. A `scipy.*` mypy override was added (scipy
ships no type stubs) so `mypy --strict` stays clean. `FitResult` gained a `factors` field so a
fitted result is self-describing in natural units (needed for natural-axis contour plots). The
plotting helpers are imported from `doe.plotting` directly (not re-exported at top level, matching
the Phase 1 `pareto_plot` / `main_effects_plot` convention).

New deps: none required. `scipy.stats` (F-distribution p-values) is already a core dep.
ANOVA is implemented **in-house** so the core install needs nothing new; `statsmodels` stays an
unused optional extra.

---

## 1. Design generators — `generators/rsm.py`

### 1.1 Central composite design (CCD)

A CCD = a 2-level factorial (or resolution-V fraction) **core** + **axial/star** points at
`±α` on each axis + replicated **center** points.

```python
def central_composite(
    factors: Sequence[Factor],
    *,
    alpha: float | Literal["faced", "rotatable", "orthogonal"] = "faced",
    center: int = 4,                 # number of center-point replicates
    fraction: Sequence[str] | None = None,  # optional generators for a fractional core
) -> Design: ...
```

Run structure for *k* continuous factors:

- **Factorial core:** `full_factorial` (or `fractional_factorial` when `fraction` given),
  coded `±1`.
- **Axial points:** `2k` runs, each factor at `±α`, all others at 0.
- **Center points:** `center` runs at the origin (all factors 0).

Axial distance `α`:

| Type        | α                                   | Property                                        |
|-------------|-------------------------------------|-------------------------------------------------|
| `faced`     | `1.0` (face-centered CCF) — **default** | all runs stay inside the stated `low`/`high` |
| `rotatable` | `(n_factorial)**0.25`               | constant prediction variance at a given radius  |
| `orthogonal`| solves for uncorrelated 2nd-order   | block-orthogonal quadratic terms                |

**Default = face-centered (`alpha="faced"`, α = 1).** This keeps every run inside the user's
stated factor bounds — the safe choice when `low`/`high` are hard physical limits, which is the
common case. The price is that the design is *not* rotatable; users who want rotatability opt in
with `alpha="rotatable"` (k=2 → α≈1.414, k=3 → α≈1.682, the test anchors below).

**Coding note:** with the face-centered default, axial points sit at coded `±1` and decode
exactly onto `low`/`high` — no extrapolation, and `ContinuousFactor.decode` handles it as-is.
When a user opts into `rotatable`/`orthogonal`, axial points fall at `±α` *outside* `[-1,+1]`
and therefore decode *beyond* the stated bounds (circumscribed CCC behaviour). We document that
clearly and emit a note in the design `meta` so it's never a silent surprise. (An inscribed
rescaling variant — pull factorial points inward so `±α` lands on the bounds — is a possible
later add, not in 2a.)

### 1.2 Box–Behnken design (BBD)

Spherical 3-level design, no corner (extreme) runs — good when corners are infeasible. Built
from balanced incomplete block pairings: for each pair of factors, run the `±1` factorial of
that pair with all other factors at 0, then add center points.

```python
def box_behnken(factors: Sequence[Factor], *, center: int = 3) -> Design: ...
```

Run-count anchors for tests: k=3 → 12 edge + 3 center = **15**; k=4 → 24 + center; k=5 → 40 +
center. Requires k ≥ 3.

### 1.3 Center points & replicates in `Design`

Center points must be *distinguishable* (they're identical rows in natural units) because pure
error for lack-of-fit comes from their replication. Add to `design.py`:

- A `point_type` column (or `meta`-tracked index) tagging each run as
  `factorial` / `axial` / `center` / `edge`.
- `Design.center_indices` / `Design.n_center` helpers.
- Keep `coded()` and `randomize()` working unchanged — `randomize` should preserve the
  `point_type` tag through the shuffle.

This is the one Phase 1 container change Phase 2 depends on; keep it additive (new optional
column, defaulted) so existing factorial designs and the 10 current tests stay green.

---

## 2. Model specification — `analysis/model.py`

`build_model_matrix(order=2, interactions=True)` already emits intercept + mains + 2-factor
interactions + squares — that *is* the full quadratic model. Phase 2a cleanups:

- Add a `model="quadratic"` convenience alongside the existing `order`/`interactions` flags so
  callers don't have to remember `order=2, interactions=True`.
- **Drop squared terms for factors that have no 3rd level** (a pure `±1` factor has `xᵢ² ≡ 1`,
  collinear with the intercept). Detect from the design (does the coded column take a value off
  `{-1,+1}`?) and skip those squares so `X` stays full rank.
- Term ordering: mains, then interactions, then squares (stable, matches ANOVA grouping).
- Categorical contrast expansion stays deferred to a later phase — document it, don't build it.

---

## 3. Fitting & inference — `analysis/fit.py` + `analysis/anova.py`

### 3.1 `FitResult` gains inference fields

`fit_ols` keeps its signature but `FitResult` grows (all additive, existing fields unchanged):

- `dof_resid: int` — `n_runs − n_terms`.
- `mse: float` — residual mean square `ss_res / dof_resid`.
- `cov_beta: np.ndarray` — `mse · (XᵀX)⁻¹`.
- `std_errors: np.ndarray` — `sqrt(diag(cov_beta))`.
- `t_values`, `p_values` — coefficient t-tests via `scipy.stats.t`.
- `conf_int(level=0.95)` method — `coef ± t_crit · se`.

Guard `dof_resid > 0` (a saturated factorial has none) — return `nan` SEs with a warning rather
than dividing by zero, so Phase 1's exact-fit tests (`r²=1`) don't break.

### 3.2 ANOVA — `analysis/anova.py` (in-house, sequential / Type I)

```python
def anova_table(result: FitResult, design: Design, response: np.ndarray) -> pd.DataFrame: ...
def lack_of_fit(result: FitResult, design: Design, response: np.ndarray) -> LackOfFit: ...
```

`anova_table` uses **sequential (Type I) sums of squares**: terms are added in model order and
each row's SS is the reduction in residual SS from adding that term to the ones before it.
Computed in-house from QR/Gram–Schmidt on `X` (numerically stabler than repeated normal-equation
refits); F and p from `scipy.stats.f`.

Rows: each model term (or grouped: linear / interaction / quadratic), residual, total. Columns:
`SS, df, MS, F, p`. (Type III / partial SS is noted as a later refinement — Type I is order-
dependent, which we document so users reading the table aren't surprised.)

`lack_of_fit` splits residual SS into:
- **Pure error** from replicated center points: `SS_PE = Σ(y_center − ȳ_center)²`, df = `n_center − 1`.
- **Lack of fit:** `SS_LOF = SS_resid − SS_PE`, df = `dof_resid − df_PE`.
- F = `MS_LOF / MS_PE`, p from `scipy.stats.f`. A *non*-significant LOF is what we want — the
  quadratic model is adequate.

### 3.3 Predicted R² (Q²) via PRESS

`PRESS = Σ (eᵢ / (1 − hᵢ))²` where `hᵢ` are leverages from the hat matrix
`H = X(XᵀX)⁻¹Xᵀ` (no refitting needed). Then `R²_pred = 1 − PRESS / SS_tot`, and
`R²_adj = 1 − (1−R²)(n−1)/(n−p)`. Add all three to the ANOVA summary.

---

## 4. Plotting — `plotting.py` (Phase 2a subset) ✅ DONE

All lazy-import `matplotlib` as today; add to the `[plotting]` extra only.

- `surface_grid(result, x, y, *, fixed=None, resolution=25)` — the headless, numerically-testable
  core: evaluates the fitted surface over a grid of two factors (others held at center or
  `fixed`, in *natural* values) and returns natural-unit mesh arrays `(X, Y, Z)`. Reconstructs
  each model column from `result.term_names` (intercept / main / `a:b` / `a^2`). `contour_plot`
  is a thin wrapper over it; this is also what Phase 2b's 3-D `surface_plot` will reuse.
- `contour_plot(result, x, y, *, fixed=None, ax=None, resolution=25, filled=True)` — filled
  `contourf` + labelled `contour` lines over `surface_grid`, axes in natural units.
- `residuals_vs_fitted(result, ax=None)` and `normal_qq(result, ax=None)` — standard diagnostics
  (`scipy.stats.probplot` for the Q–Q line).
- `half_normal_plot(result, ax=None)` — effects vs half-normal quantiles, the screening
  companion to the existing `pareto_plot` (backfills a Phase 1 gap).

Drawing contours in natural units required `FitResult` to know its factors, so a `factors:
FactorSet` field was added to `FitResult` (set by `fit_ols`). `tests/conftest.py` forces the Agg
backend so the plotting tests run headless.

3-D `surface_plot` shipped with Phase 2b (§8) alongside the optimization it visualizes; it
reuses this `surface_grid` core.

---

## 5. Public API additions (`__init__.py`, Phase 2a)

Add: `central_composite`, `box_behnken` (generators); `anova_table`, `lack_of_fit` (analysis).
Keep the flat top-level namespace established in Phase 1.

---

## 6. Tests (anchors, Phase 2a)

- **`test_rsm.py`** — CCD default (faced) k=2/k=3: all coded values in `{−1,0,1}`, runs decode
  on/inside bounds, run count = factorial + 2k axial + center. CCD `alpha="rotatable"`: α ≈
  1.414 (k=2) / 1.682 (k=3), axial points decode beyond bounds. Box–Behnken k=3 = 15 runs, no
  corner points, all factors ∈ {−1,0,1}. Center points decode to factor centers.
- **`test_anova.py`** — fit a known quadratic `y = 50 + 3x₁ − 2x₂ + 4x₁² − x₁x₂` on a CCD and
  recover coefficients to tolerance; sequential SS sums to the model SS; LOF non-significant when
  the true model is quadratic and *significant* when fitting a linear model to curved data;
  R²_pred < R² always; pure-error df = `n_center − 1`.

Run gates unchanged: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.

---

## 7. Build order (Phase 2a)

1. ✅ `generators/rsm.py` + center-point support in `design.py` + `test_rsm.py` (designs first —
   everything downstream consumes them).
2. ✅ Model-matrix quadratic cleanup + standard errors/CIs in `fit.py`.
3. ✅ `analysis/anova.py` (sequential ANOVA, lack-of-fit, PRESS) + `test_anova.py`.
4. ✅ `surface_grid` + contour plot + residual/half-normal diagnostics in `plotting.py` +
   `test_plotting.py` (test-first).

All four steps shipped test-first; the suite is green (54 tests, ruff + mypy clean). Phase 2a is
complete — the next roadmap work is Phase 2b (§8), which is deferred.

**Three API decisions confirmed during the build:**
- Center-point tracking lives on `Design` (`point_types` tuple + `n_center` / `center_indices`),
  not as a `point_type` column in `runs` — keeps `runs` pure factor/response data.
- `lack_of_fit` returns a `LackOfFit` dataclass rather than appending a row to the ANOVA table.
- `surface_grid` (headless natural-unit mesh evaluator) is the testable core under `contour_plot`,
  and `FitResult` gained a `factors` field so results are self-describing in natural units.

---

## 8. Phase 2b — surface optimization ✅ DONE

Built after 2a shipped. All items below are implemented (`analysis/optimize.py`,
`tests/test_optimize.py`, `surface_plot` in `plotting.py`):

- ✅ **`analysis/optimize.py`** — writes the fitted quadratic as `ŷ = b₀ + xᵀb + xᵀB x`
  (`_quadratic_form` extracts `b` and the symmetric `B` from `FitResult.term_names`, using
  `B[i,i] = coef(xᵢ²)` and `B[i,j] = ½·coef(xᵢ:xⱼ)`).
  - ✅ `stationary_point` — `x_s = −½ B⁻¹ b`, decoded to natural units, with canonical analysis
    (eigendecomposition of `B` via `eigh` → `"maximum"` / `"minimum"` / `"saddle"`). Raises when
    `B` is rank-deficient (no unique stationary point).
  - ✅ `optimum` — constrained optimum over the coded box via multistart `scipy.optimize`
    (`L-BFGS-B`), reporting whether the optimum sits on a bound (`at_bound`). `bounds` accepts a
    single `(low, high)` or a per-factor sequence.
  - ✅ Multi-response **desirability** (Derringer–Suich): `ResponseGoal` (`max`/`min`/`target`,
    `low`/`high`/`target`/`weight`) → per-response `dᵢ(ŷ)`, geometric-mean `D = (∏dᵢ)^(1/m)`,
    maximized over the box with `differential_evolution` (the desirabilities are non-smooth).
- ✅ **`plotting.py`** — `surface_plot` (3-D `plot_surface`) over the existing `surface_grid` core,
  companion to the 2a contour plot.
- ✅ **API** — `stationary_point`, `optimum`, `desirability` (plus the `StationaryPoint`,
  `Optimum`, `ResponseGoal`, `DesirabilityResult` dataclasses) exported at top level.
- ✅ **Tests** (`test_optimize.py`) — known concave quadratic: stationary point matches the
  closed-form `−½ B⁻¹ b` and canonical analysis reports a maximum; convex → minimum, mixed →
  saddle; constrained optimum clamps to the box when the stationary point is outside it; a
  conflicting two-response desirability trades off at the analytic compromise (`x₁ = 0.5`, `D = 0.5`).

**API decisions confirmed during the 2b build:**
- The rank check uses an absolute tolerance tied to the coefficient scale, so a purely linear
  fit's ~1e-16 interaction round-off reads as "no curvature" rather than a spurious full rank.
- `desirability` uses `differential_evolution` (global, gradient-free) because the saturated
  flats of each `dᵢ` make the objective non-smooth; `optimum` uses smooth multistart `L-BFGS-B`.

---

## 9. Resolved decisions

- **Default CCD coding** — *face-centered* (α=1, all runs inside stated bounds);
  `rotatable`/`orthogonal` are opt-in and documented as extrapolating beyond bounds. ✅
- **ANOVA SS type** — *in-house, sequential (Type I)*; no new required deps; Type III noted as a
  later refinement. ✅
- **Phase scope** — *trimmed*: 2a = designs + fitting + ANOVA + contour plots; optimization &
  desirability deferred to 2b. ✅
- **Center-point tracking** — lives on `Design` (`point_types` + `n_center` / `center_indices`),
  not a column in `runs`. ✅
- **Lack-of-fit return** — a `LackOfFit` dataclass, not a row appended to the ANOVA table. ✅
- **Surface evaluation** — a public `surface_grid` helper (headless, natural-unit `(X, Y, Z)`
  mesh) is the numerically-tested core; `contour_plot` is a thin wrapper. To support natural-unit
  axes, `FitResult` carries its `FactorSet` via a new `factors` field. ✅
