# Phase 5 — Screening & Restricted Randomization

Detailed build plan for Phase 5. Phases 1–4 covered factorial screening, response-surface
methodology, optimization, diagnostics, computer-generated optimal designs, space-filling
sampling, and mixture designs — every family assumed the runs could be executed in **fully
randomized order** and fitted with **ordinary least squares**. Phase 5 adds the mainstream
families `PLAN.md` §2 gestures at but has not planned in detail, all sharing the theme of
*efficient screening* and *randomization that reality constrains*:

1. **Definitive screening designs (DSD)** — Jones–Nachtsheim conference-matrix designs that
   screen main effects *and* detect curvature / two-factor interactions in `2k + 1` runs,
   collapsing the classic full-factorial → CCD two-stage screen into one design.
2. **Split-plot / hard-to-change factors** — restricted randomization for factors that cannot
   be reset every run (oven temperature, furnace atmosphere, a reagent lot). This is the first
   family whose runs are *not* freely randomizable, and it is the first analysis-layer
   extension since the Scheffé no-intercept path: two error strata mean a GLS/REML fit, not OLS.
3. **Classical / blocking** — randomized complete block designs, Latin squares, and blocked
   factorials (fractions assigned to blocks via defining contrasts), plus richer run-order
   utilities.

The Phase 1–4 contract is preserved where it applies: generators return a `Design`, analysis
consumes `Design.coded()` → model matrix → fit. **DSD fits that contract unchanged** — it is a
generator plus a documented analysis *strategy*, no new fit path. **Blocking** adds a block
column that the existing categorical-expansion machinery already knows how to absorb.
**Split-plot** is the deliberate extension: the design carries whole-plot structure and the fit
estimates two variance components, so it is the one piece that reaches into `design.py` *and*
`analysis/fit.py`.

> **Scope decision:** Phase 5 splits into three independent sub-phases by how far each reaches
> into the stack. **Phase 5a** = DSD (`generators/screening.py` + tests + a vignette) — small,
> self-contained, everything downstream already works; ship it first. **Phase 5b** = split-plot
> (whole-plot structure on `Design`, a `fit_gls`/REML path in `analysis/`) — the larger piece,
> because it touches the modeling layer. **Phase 5c** = classical/blocking
> (`generators/blocking.py` + run-order utilities) — generator-side, low-risk, orthogonal to
> the other two. 5a/5b/5c are independent; build 5a first for an early, low-risk ship.

> **Scope decision (split-plot):** Phase 5b supports **two strata only** — one whole-plot factor
> group and one sub-plot factor group, the classic industrial split-plot. Split-split-plots
> (three+ strata) and strip-plot / split-block designs are well-defined extensions but multiply
> the covariance structures; they are deferred, not designed around.

## Goals

By the end of Phase 5 a user can:

1. Generate a **definitive screening design** over any set of continuous factors (optionally
   with a few two-level categoricals via the Jones–Nachtsheim extension), get a `2k + 1`-run
   three-level design that is orthogonal for main effects and lets them estimate curvature, and
   fit it with the existing `fit_ols`/`anova_table`/`half_normal_plot` machinery unchanged.
2. Declare factors **hard-to-change**, generate a **split-plot** design that groups runs into
   whole plots (hard-to-change factors held constant within a plot, easy-to-change factors
   randomized inside it), and fit it with **REML/GLS** that separates whole-plot from sub-plot
   error — recovering correct standard errors that OLS would understate for whole-plot effects.
3. Generate **classical blocked designs** — randomized complete block, Latin square, and
   blocked `2^(k-p)` factorials — with the block effect carried as a design column the model
   matrix absorbs, and use richer **run-order** utilities (block-aware randomization, restart
   grouping) built on today's `Design.randomize`.

Correctness anchors (continuing the Montgomery/Jones–Nachtsheim/Cornell pattern): a DSD on
`k` factors has exactly `2k + 1` runs, three levels `{−1, 0, +1}`, a main-effects information
matrix that is diagonal (main effects orthogonal to each other and to quadratic terms), and two
zero entries per factor column plus one all-zero center run; the `k = 6` DSD reproduces the
13-run design tabulated in Jones & Nachtsheim (2011). A split-plot REML fit reproduces the
variance-component estimates and whole-plot/sub-plot standard errors of a textbook example
(Montgomery §14, the plasma-etch or a comparable published split-plot); OLS on the same data
under-reports the whole-plot standard error, which the test asserts. A randomized complete block
design has every treatment exactly once per block; a `k×k` Latin square has each treatment once
per row and once per column.

## What's new / what changes

```
src/doe/
  design.py           # CHANGED: optional whole_plots field (tuple[int, ...] | None) tagging
                      #          each run's whole plot; carried through replicate/randomize/
                      #          project/to_dict/from_dict like point_types. New blocks field
                      #          (or reuse a "block" column convention — decided in §3).
  factors.py          # CHANGED: Factor gains a `hard_to_change` flag (continuous/categorical);
                      #          FactorSet exposes whole_plot_factors / sub_plot_factors helpers
  generators/
    screening.py      # NEW: definitive_screening (conference-matrix DSD, +categorical extension)
    splitplot.py      # NEW: split_plot (whole-plot × sub-plot crossing + restricted randomize)
    blocking.py       # NEW: randomized_complete_block, latin_square, blocked_factorial
  analysis/
    fit.py            # CHANGED: fit_gls (REML two-stratum fit) sharing FitResult;
                      #          fit_ols untouched. ModelSpec unchanged (DSD uses "quadratic").
    variance.py       # NEW: REML variance-component estimation (whole-plot + residual strata)
  serialization.py    # CHANGED: schema accepts whole_plots / blocks and hard_to_change
  __init__.py         # exports the new public surface (see §4)
tests/
  test_screening.py   # NEW (DSD)
  test_splitplot.py   # NEW (generation + REML fit)
  test_blocking.py    # NEW
docs/
  VIGNETTES.md        # up to three new vignettes (see §6); assets via build_vignette_assets.py
  SERIALIZATION.md    # CHANGED: document whole_plots / blocks / hard_to_change fields
```

New deps: **none required.** Conference-matrix construction, block enumeration, and Latin
squares are pure numpy/itertools. REML is a small `scipy.optimize.minimize` over one or two
variance-ratio parameters with closed-form GLS at each step — no `statsmodels` needed (it stays
the optional extra it is today, not a hard dependency).

---

## 1. Definitive screening designs — `generators/screening.py` (Phase 5a)

DSDs answer the single most common industrial screening question — "which of my `k` factors
matter, and is any relationship curved?" — in `2k + 1` runs, without the full-factorial → CCD
two-stage detour. Their defining properties (Jones & Nachtsheim 2011): main effects are
orthogonal to each other and to *all* second-order terms, so a main effect estimate is never
biased by an active two-factor interaction or quadratic; every factor is run at three levels, so
curvature is estimable; and no two-factor interaction is fully aliased with any other effect.

### 1.1 Generator

```python
def definitive_screening(
    factors: Sequence[Factor], *,
    extra_center_runs: int = 0,
    fake_factors: int | None = None,   # None → auto (add one iff k is odd)
) -> Design: ...
```

- **Construction.** For an even number of continuous factors `k`, build a conference matrix
  `C` of order `k` (`C` square, zero diagonal, `±1` off-diagonal, `CᵀC = (k−1)I`). The design is
  the row stack `[C; −C; 0ᵀ]` — the foldover of `C` plus one all-zero center run — giving
  `2k + 1` three-level runs in coded units. For odd `k`, add one **fake factor** to reach even
  order, generate, and drop the fake column (yields `2k + 3` runs); `fake_factors` lets the user
  override, `None` auto-adds the minimum. Conference matrices for the small orders are built from
  the tabulated seeds (Paley/symmetric constructions for `k ≤ 30`), the range that covers every
  realistic screening problem; an unconstructible order raises with the nearest valid `k`.
- **Levels & decode.** Coded `{−1, 0, +1}` map to low / midpoint / high in natural units via the
  existing `ContinuousFactor.decode` — stored in natural units like every generator. The midpoint
  level is exactly why squared terms become estimable, and `build_model_matrix` already emits a
  squared column for any continuous factor taking values off `{−1, +1}` (Phase 2 behaviour), so
  **no model-matrix change is needed**.
- **Categorical extension.** Two-level categorical factors are supported via the Jones–Nachtsheim
  (2013) DSD-with-categoricals construction (categoricals enter at `±1`, no midpoint); >2-level
  categoricals raise, pointing to `d_optimal` as the general-purpose alternative. Continuous-only
  is the common path and the first to land.
- **`point_types`** tags the all-zero run `"center"` (plus any `extra_center_runs`), so
  `n_center` / `lack_of_fit` work exactly as they do for a CCD. `meta["generator"]` records the
  call (via the existing `_generator_spec` helper) for regeneration and serialization.

### 1.2 Analysis is unchanged — but the *strategy* is documented

A DSD has `2k + 1` runs and a full quadratic model has `1 + 2k + C(k,2)` terms, so the quadratic
model is **saturated or supersaturated** — you cannot fit it all at once. This is a strategy
question, not a code change: the vignette (§6) demonstrates the standard DSD analysis flow using
existing machinery — fit main effects with `fit_ols(model="linear")`, read the `half_normal_plot`
to find the active few, then fit a reduced quadratic in only the active factors (where residual
dof is now positive) and run `anova_table`. No new analysis function is required; the value of
DSD is entirely in the design, which the existing fit path already consumes.

Note `lack_of_fit` is *not* part of the default flow: a default DSD has a single center run, so
the pure-error estimate has zero degrees of freedom and `lack_of_fit` raises. To use it, pass
`extra_center_runs >= 1` (giving ≥ 2 center runs) — the analog of adding center replicates to a
CCD. The vignette (§6) therefore stops at `anova_table`.

### 1.3 As-built notes & follow-ups (post-implementation code review)

Phase 5a shipped as `generators/screening.py` (`definitive_screening`) + `tests/test_screening.py`
+ the Vignette 21 walkthrough. A high-effort code review found **no correctness bugs** — the
conference-matrix construction is verified (`CᵀC = (k−1)I`, zero diagonal, `±1` off-diagonal) for
every supported order, the DSD orthogonality/level/center invariants hold, and a DSD round-trips
through serialization unchanged. Two places where the shipped code diverges from the plan above,
plus the review's robustness/quality follow-ups, are tracked here so the plan stays truthful.

**As-built deltas from §1.1–§1.2:**

- **Conference matrices are constructed, not tabulated.** §1.1 says "tabulated seeds
  (Paley/symmetric constructions for `k ≤ 30`)". The implementation instead uses a general **Paley
  border construction** (`C = [[0, 1ᵀ], [1, Q]]`, `Q` the Jacobsthal/quadratic-character matrix)
  over `GF(p^m)`, plus a **skew doubling** construction. Together they cover **every even order
  from 2 to 32 except 22** — and order 22 provably does not exist (§1.4.1). `definitive_screening`
  also **auto-advances `fake_factors` to the next constructible order** (follow-up 1, done), which
  now only ever bites at `k = 21`/`k = 22` (advancing to order 24).
- **The categorical extension is not implemented.** §1.1 describes the Jones–Nachtsheim (2013)
  two-level categorical DSD; the shipped code **rejects all categorical factors** (pointing to
  `d_optimal`). Continuous-only landed first, as planned; the categorical path is deferred
  (build plan in §1.4.2).

**Follow-ups (ranked; all implemented — the open items, the missing conference-matrix orders
and the categorical extension, are planned in §1.4):**

1. **Close the coverage gap / search forward for a constructible order.** *(Done, both ways.)*
   With `fake_factors=None`, `definitive_screening` auto-advances `n_fake` by 2 until
   `_constructible_order(k + n_fake)` holds. The construction itself has since been widened
   (§1.4.1, done), so orders 16 and 28 now build directly and `k = 16`/`k = 28` get their *exact*
   `2k + 1` designs; the fallback advance now only fires around the nonexistent order 22.
2. **Make the unconstructible-order error actionable.** *(Done.)* An explicit `fake_factors` that
   leaves an unconstructible order now raises in terms of `k`, states the shortfall, and suggests
   concrete `fake_factors` values (with their run counts) via `_suggest_fake_factors`.
3. **Reconcile the default with the documented `lack_of_fit` flow.** *(Done — docs.)* §1.2 now
   states `lack_of_fit` is not in the default flow (one center run ⇒ zero pure-error dof) and that
   `extra_center_runs >= 1` is required to use it; the default run count is unchanged (keeps the
   `2k + 1` property intact) and the vignette stops at `anova_table`.
4. **Fix the mixture-rejection message.** *(Done.)* `definitive_screening` no longer routes
   mixture rejection through `factorial._require_box_factors`; it checks `MixtureFactor` directly
   and raises its own message ("a DSD screens a box region, not the simplex").
5. **Minor cleanups.** *(Done.)* `_conference_matrix` now calls `_constructible_order` (with an
   `assert factor is not None`) instead of re-deriving the predicate; the `m = 1` Jacobsthal matrix
   is built as `scipy.linalg.circulant` of the character vector (the matrix is circulant because
   `Q[i, j] = chi(i − j mod p)`), matching how `factorial.py` reuses `toeplitz`/`hankel`; the
   unreachable `m > 2` branch carries `# pragma: no cover`; and the k=6 test is renamed
   `test_dsd_k6_main_effects_independent_of_all_second_order` and now asserts the DSD-defining
   property that main effects are orthogonal to *every* two-factor interaction as well as every
   quadratic term — strictly stronger than, and no longer a duplicate of,
   `test_dsd_main_effects_orthogonal`.

### 1.4 Finishing Phase 5a — build plan for the deferred work

Two items remained from §1.1–§1.3: the missing conference-matrix orders (§1.4.1, **done**) and
the Jones–Nachtsheim (2013) two-level categorical extension (§1.4.2, still open). They are
independent, but the orders were built first: the categorical construction consumes a conference
matrix of order `m + c`, so every order closed widens the categorical coverage for free.

#### 1.4.1 Conference-matrix coverage — orders 16 and 28 (order 22 does not exist) — *done*

> **As built.** Implemented as planned with one simplification: **order 28 needed no skew
> construction at all.** The plan assumed the skew Paley border was required for `q ≡ 3 (mod 4)`,
> but `Q Qᵀ = q I − J` holds for *every* prime power regardless of the character's symmetry, so
> the existing all-`+1` border already yields a valid (non-skew) conference matrix over `GF(27)`
> — the shipped orders 4, 8 and 12 (`q ≡ 3 mod 4`) were always being built that way. Order 28
> therefore reduced to step 3 alone (the `GF(p³)` Jacobsthal matrix). The skew border (step 1)
> is still needed, but *only* to seed the doubling that reaches order 16, so it lives in a
> dedicated `_skew_conference_matrix` rather than as a variant of the main builder.
>
> Coverage is now every even order 2–32 except 22. `k = 16` builds the exact 33-run DSD (was 37
> runs at order 18) and `k = 15` lands on order 16; `k = 28` is exact and `k = 27` takes one fake
> factor. `k = 21`/`k = 22` still advance to order 24, and forcing order 22 raises a message
> saying the matrix *does not exist* (it would have to be symmetric, requiring 21 to be a sum of
> two squares) rather than merely being unimplemented — `_order_exists` encodes that distinction.

Existence facts that bound the work (Belevitch / van Lint–Seidel):

- For `n ≡ 2 (mod 4)` a conference matrix can exist only in *symmetric* form, which requires
  `n − 1` to be a sum of two squares. `21 = 3·7` is not, so **no conference matrix of order 22
  exists** — `k = 21` is permanently served by the fake-factor advance to order 24. This is a
  mathematical fact to document in the error message, not a gap to close.
- For `n ≡ 0 (mod 4)` conference matrices are *skew* (`Cᵀ = −C`). Orders 16 and 28 both exist
  in skew form; they are the real gaps. A skew `C` still satisfies `CᵀC = (n−1)I` with zero
  diagonal and `±1` off-diagonal, which is all the DSD fold-over `[C; −C; 0]` uses — the DSD
  invariant tests apply unchanged.

Construction plan (pure numpy, no new deps; each step is independently testable):

1. **Skew Paley form.** For prime-power `q ≡ 3 (mod 4)` the Jacobsthal matrix is already skew
   (`Qᵀ = −Q`, since `χ(−1) = −1`); bordering it as `[[0, 1ᵀ], [−1, Q]]` (note the `−1` column,
   vs. the current all-`+1` border) yields a *skew* conference matrix of order `q + 1`. Add this
   as a skew variant of `_conference_matrix` — the existing non-skew output stays as-is for the
   orders already served, so shipped designs are bit-stable.
2. **Skew doubling closes order 16.** If `C` is a skew conference matrix of order `n`, then
   `[[C, C+I], [C−I, −C]]` is a skew conference matrix of order `2n` (blockwise check using
   `Cᵀ = −C` and `CᵀC = (n−1)I`; the identity terms put `±1` where the off-diagonal blocks need
   them and the diagonal blocks keep the zero diagonal). Order 8 (`q = 7`, skew via step 1)
   doubles to 16. The doubling requires a *skew* input — doubling the symmetric order-14 matrix
   does **not** give 28.
3. **`GF(p³)` Jacobsthal closes order 28.** `q = 27 = 3³ ≡ 3 (mod 4)` is a prime *cube*, one
   step past the current `GF(p)`/`GF(p²)` support. Generalize `_jacobsthal_matrix` to `GF(p^m)`:
   elements as coefficient tuples multiplied modulo an irreducible polynomial (e.g.
   `x³ − x + 1` over `GF(3)` — no roots, hence irreducible), and `χ` computed by enumerating
   the nonzero squares rather than exponentiating. The `m = 2` code is already most of this;
   whether to generalize fully or special-case `m = 3` is a build-time call (smallest clear
   diff wins). Then order 28 is the skew Paley border of step 1 over `GF(27)`.
4. **Wire up and re-describe.** `_constructible_order` learns the new orders (result: every
   even order 4–30 *except 22* is constructible); the `# pragma: no cover` on the old `m > 2`
   branch goes away; the explicit-`fake_factors` error and `_suggest_fake_factors` output for
   `k = 21` state that order 22 is nonexistent (not merely unimplemented); the §1.3 delta text
   above is updated to match.

Anchors (`test_screening.py`):

- The parametrized conference-matrix invariant test (`CᵀC = (n−1)I`, zero diagonal, `±1`
  off-diagonal) extends to orders 16 and 28; the skew constructions additionally assert
  `Cᵀ = −C`.
- `k = 16` now builds the *exact* 33-run DSD with no fake factors (previously auto-advanced to
  order 18 / 37 runs); `k = 15` lands on order 16; `k = 27` (order 28 via one fake) and
  `k = 28` (exact) build.
- `k = 21` still auto-advances to order 24, and an explicit `fake_factors=0` raises a message
  mentioning nonexistence.

#### 1.4.2 Categorical extension — Jones–Nachtsheim (2013) DSD-augment — *done*

> **As built.** Implemented as `_dsd_augment_categorical` in `generators/screening.py`;
> `definitive_screening` now dispatches to it whenever any factor is a (two-level)
> `CategoricalFactor`, and the all-continuous path is untouched (regression-tested bit-identical
> against a frozen `k = 4` design). The construction follows the paper: build `[C; −C]` over the
> conference matrix of order `m + c` (existing fake-factor logic on `k = m + c`), assign the last
> `c` kept columns to the categoricals, replace each categorical column's zero pair with
> `z_j` / `−z_j`, and append one pseudo-center pair (continuous at 0, categorical `j` at
> `±b_j`) — so `n = 2·order + 2` runs (`n₁ + 2`). `z, b ∈ {±1}^c` are chosen by exhaustively
> maximizing `det(X₁ᵀX₁)` over all `2^(2c)` combinations in a deterministic iteration order
> (`2c > 20` raises, pointing at `d_optimal`). `point_types` tags the pair `"pseudo-center"` (not
> `"center"`); `extra_center_runs` appends whole replicated pseudo-center *pairs*;
> `meta["categorical_signs"]` records the resolved `z`/`b`.
>
> **One deviation from the sketch/eq. (2).** The paper's Table-2/eq.-(2) example for `m = 4,
> c = 2` reports `det(X₁ᵀX₁) = 20 428 800`, but that value is **not achievable** with an order-6
> conference matrix (which is unique up to equivalence): the exhaustive `2^(2c)` search yields
> `{16 486 400, 20 966 400}`, so our D-optimal augmentation lands at **20 966 400** — a weakly
> *higher*-determinant design than the paper's tabulated instance, with the *same* diagonal
> `(14, 10, 10, 10, 10, 14, 14)` and off-diagonals in `{0, ±2}`. The definitive property (the
> real anchor) holds exactly: the alias matrix's main-effect rows against every quadratic and
> two-factor-interaction column are `0` to machine precision. The run counts reproduce Table 4's
> `n_DSD` column (m=4,c=1→14; m=4,c=2→14; m=4,c=3/4→18; …). The tests assert structure + the
> definitive property + Table-4 run counts rather than the specific eq.-(2) determinant.

> **Scope decision:** implement **DSD-augment only**. Of the paper's two methods, DSD-augment
> keeps the property that names the class — every main effect remains unbiased by any active
> second-order effect — at the cost of slightly non-orthogonal main effects; ORTH-augment gains
> exact main-effect orthogonality but introduces partial aliasing between main effects and
> interactions involving the categoricals, surrendering the definitive property. No `method=`
> kwarg until a second method actually ships.

**Step 0 (gates the rest): obtain the paper** (*JQT* 45(2), 121–129) and transcribe the exact
augmentation rules — the sign conventions for the replaced zeros and the pseudo-center pair —
plus its tabulated example designs, which become test fixtures. The outline below is the shape
of the construction; the paper is the authority on the sign choices.

Construction (`m` continuous + `c` two-level categorical, `k = m + c`):

1. Build the conference matrix of order `k` (the §1.1 fake-factor logic operates on `k`
   unchanged, now with §1.4.1's wider coverage).
2. Stack `[C; −C]`; assign `m` columns to the continuous factors (levels `{−1, 0, +1}`, decoded
   via `ContinuousFactor.decode` as today) and `c` columns to the categoricals.
3. A categorical column cannot hold `0`: replace each zero pair (the diagonal zero in `C` and
   its fold-over image in `−C`) with `±1` per the paper's rule.
4. Replace the single all-zero center run with a **pseudo-center pair**: continuous factors at
   `0`, categoricals at all-`−1` and all-`+1` — giving `2(m + c) + 2` runs.
5. Decode categoricals as `−1` → first level, `+1` → second level, matching the deviation-coding
   convention `build_model_matrix` already uses, so the fitted categorical contrast column
   reproduces the design column exactly.

Semantics to get right (decided now, not in the code review):

- **`point_types` tags the pair `"pseudo-center"`, not `"center"`.** The two runs differ in
  their categorical coordinates, so they are *not* replicates; tagging them `"center"` would
  hand `lack_of_fit` a false pure-error estimate. Consequently `lack_of_fit` stays out of the
  categorical-DSD flow entirely — a documented limitation. (Generalizing pure error to "any
  replicated setting" would lift it, but that is an `anova.py` follow-up, not part of 5a.)
- **`extra_center_runs` with `c ≥ 1` adds whole pseudo-center *pairs*** (continuous at 0,
  categoricals all-`−1`/all-`+1`), keeping the two categorical arms balanced.
- **Validation:** two-level `CategoricalFactor`s are accepted; >2-level still raises pointing
  to `d_optimal`; mixture factors still raise; the all-continuous path must be bit-identical to
  today's output (regression-tested against a frozen design).

Anchors (`test_screening.py`):

- Reproduce a tabulated JN 2013 design from the step-0 fixtures, up to row/column/sign
  conventions — the same anchoring pattern as the `k = 6` continuous DSD.
- Structure: run count `2(m+c) + 2`; each continuous column has exactly two zeros; categorical
  columns are zero-free and balanced; the pseudo-center pair is tagged and round-trips through
  `replicate`/`randomize`/`project`/`to_dict`/`from_dict`.
- **The definitive property, asserted numerically:** for the main-effects model, the alias
  matrix `(X₁ᵀX₁)⁻¹X₁ᵀX₂` against all quadratic and two-factor-interaction columns has
  (near-)zero main-effect rows, to the tolerance the paper claims.
- Recovery: an injected response with active continuous mains, one quadratic, and one
  categorical main effect is recovered by the documented fit-reduced-model flow.

#### 1.4.3 Delivery

No new public API (`definitive_screening`'s signature is unchanged; it just stops rejecting
two-level categoricals) and no serialization schema change. Closing steps: update the §1.3
as-built deltas and Vignette 21 (optionally extending it with a categorical factor — re-run
`scripts/build_vignette_assets.py`); check whether `doe-service`'s DSD endpoint validation
rejects categorical factors and relax it with a golden contract test if so; keep
`uv run pytest` / `ruff check .` / `mypy` green in both packages throughout.

**Build order:** §1.4.1 steps 1–4 (each green before the next) → §1.4.2 step 0 (the paper) →
§1.4.2 construction + anchors → §1.4.3 docs/service sweep.

---

## 2. Split-plot / hard-to-change factors — Phase 5b — *done*

> **As built.** `ContinuousFactor`/`CategoricalFactor` gained `hard_to_change: bool = False`
> (emitted in `to_dict` only when `True`); `FactorSet` gained `whole_plot_factors` /
> `sub_plot_factors`. `Design` gained `whole_plots: tuple[int, ...] | None` (validated, carried
> through `with_response`/`project`/`replicate`/`randomize`/`to_dict`/`from_dict`, with
> `n_whole_plots`/`whole_plot_indices`), and `randomize` became plot-aware plus gained a
> `within=` keyword (block-aware); both share one `_shuffle_within_groups` helper. `replicate`
> offsets plot ids so replicated plots are new plots. `generators/splitplot.py::split_plot`
> crosses whole-plot × sub-plot designs (component designs are `"full"` or a ready-made `Design`).
> `analysis/variance.py` does one-parameter REML (`v0_inverse` block-diagonal Sherman-Morrison,
> profiled over `log η`); `analysis/fit.py::fit_gls` returns the same `FitResult` with new
> optional `sigma2_wp`/`n_whole_plots`/`dof_terms` fields and two-stratum containment df.
> `fit_ols`/`FitResult.to_dict` are untouched. Anchors: REML == closed-form balanced-split-plot
> ANOVA components (exact), simulation recovery, and the OLS-understates-whole-plot-SE trap.

The fully-randomized assumption behind every design so far is an industrial fiction whenever a
factor is expensive or slow to reset (furnace temperature, a coating bath, a raw-material lot).
The practical answer is a **split-plot**: hold the hard-to-change (whole-plot) factors fixed
across a group of runs — a *whole plot* — and randomize the easy-to-change (sub-plot) factors
*within* each whole plot. This restricts randomization, which induces a two-stratum error
structure: whole-plot effects are tested against whole-plot-to-whole-plot variation, sub-plot
effects against the smaller within-plot variation. OLS ignores this and reports one pooled error,
**understating** whole-plot standard errors (anticonservative) — the classic split-plot trap.
Phase 5b changes both the design (§2.1–2.2) and the fit (§2.3) to get it right.

### 2.1 Whole-plot structure on `Design` — `design.py` + `factors.py`

- **`Factor.hard_to_change: bool = False`** on `ContinuousFactor` / `CategoricalFactor` — the
  declaration that a factor cannot be reset per run. `FactorSet` exposes `whole_plot_factors` /
  `sub_plot_factors` partitions (mirrors the existing `is_mixture` property style).
- **`Design.whole_plots: tuple[int, ...] | None`** — a new optional field (dataclass field
  alongside `point_types`) assigning each run to a whole plot by integer id. `None` means fully
  randomized (every existing design), so the field is backward-compatible and defaulted. It is
  carried through `replicate` / `randomize` / `project` / `to_dict` / `from_dict` exactly like
  `point_types` is today, and `randomize` becomes **plot-aware**: it randomizes whole-plot order
  and sub-plot order *within* each plot, never breaking the restriction. A `n_whole_plots`
  property and `whole_plot_indices` accessor mirror `n_center` / `center_indices`.

### 2.2 Generator — `generators/splitplot.py`

```python
def split_plot(
    factors: Sequence[Factor], *,     # hard_to_change flags read off the factors
    whole_plot_design: str | Design = "full",   # design on the WP factors
    sub_plot_design: str | Design = "full",      # design on the SP factors, per whole plot
    n_whole_plot_reps: int = 1,
    seed: int | None = None,
) -> Design: ...
```

- Crosses a design on the whole-plot factors with a design on the sub-plot factors: each
  whole-plot setting is one whole plot, and the full sub-plot design is run inside it. The
  component designs can be named (`"full"`/`"fractional"`/…) or passed as ready-made `Design`
  objects generated by any Phase 1–4 generator, so a split-plot CCD or a split-plot fractional
  factorial both fall out by composition rather than a bespoke generator per combination.
- Sets `whole_plots` from the crossing and tags `point_types` where the sub-plot design has
  center points. `randomize(seed=…)` (plot-aware, §2.1) produces the execution order.
- Validates that at least one factor is `hard_to_change` and at least one is not (else it is an
  ordinary design and the generator says so), matching the fail-loud validation pattern in
  `generators/rsm.py`.

### 2.3 REML / GLS fit — `analysis/variance.py` + `analysis/fit.py`

The split-plot model is a mixed model `y = Xβ + Zγ + ε` with whole-plot random effects
`γ ~ N(0, σ²_wp I)` and sub-plot error `ε ~ N(0, σ² I)`; `Z` is the whole-plot indicator from
`Design.whole_plots`. The response covariance is block-diagonal,
`V = σ² I + σ²_wp ZZᵀ`, a compound-symmetric block per whole plot governed by the single ratio
`η = σ²_wp / σ²`.

Plan:

- **`variance.py`** — REML estimation of `η` (hence `σ²_wp`, `σ²`) by maximizing the REML
  log-likelihood over the one ratio parameter with `scipy.optimize.minimize_scalar` /
  `minimize`. `V⁻¹` is cheap because it is block-diagonal (invert each whole-plot block, or use
  the Sherman–Morrison–Woodbury form per plot). Pure numpy/scipy, headless, independently
  testable — the `diagnostics.py` pattern.
- **`fit_gls`** in `fit.py` — the GLS front door mirroring `fit_ols`: build the coded model
  matrix (unchanged `build_model_matrix`), estimate `η` via REML, then
  `β̂ = (XᵀV⁻¹X)⁻¹ XᵀV⁻¹y` with `Cov(β̂) = (XᵀV⁻¹X)⁻¹`. It returns the **same `FitResult`
  type** (coefficients, std errors, t/p, `conf_int`, R²) so every downstream consumer —
  `anova_table` (whole-plot vs sub-plot strata), the effect and diagnostic plots — reads it
  unchanged; the `FitResult` gains variance-component fields (`sigma2_wp`, `sigma2`,
  `n_whole_plots`) and a `strata` note. `fit_ols` is **untouched**; `fit_gls` is opt-in and
  requires `design.whole_plots is not None` (raises otherwise, symmetric to the mixture/model
  validation in `fit_ols`). Effects (the `2×coef` swing) keep their meaning here — unlike the
  Scheffé path — because split-plot factors are still coded to `[−1, +1]`.
- **Degrees of freedom** follow the two strata: whole-plot effects get whole-plot dof, sub-plot
  effects the residual dof. The ANOVA convention is documented once and tested against the
  textbook example rather than re-derived.

---

## 3. Classical / blocking — `generators/blocking.py` (Phase 5c) — *done*

> **As built.** `generators/blocking.py` ships `randomized_complete_block`, `latin_square`, and
> `blocked_factorial`, using **block representation (b)**: a reserved `CategoricalFactor("block",
> ("B1", …))` in the `FactorSet`, so `build_model_matrix`'s deviation coding fits the block with
> zero analysis changes. Within-block run order uses the shared `Design.randomize(within="block")`
> machinery from §2. `blocked_factorial` records the confounded generators **and** their
> generalized interactions in `meta["confounded_with_blocks"]` (e.g. `["ABC", "BCD", "AD"]`).
> Anchors: RCB(4,3)→12 runs each treatment once/block; `latin_square(5)`→25 runs each treatment
> once per row and column; the block column equals the confounded contrast numerically, that
> effect is inestimable, and the rest fit clean.

Blocking removes a known nuisance source (day, batch, operator) by grouping runs so the nuisance
is constant within a block and the treatment comparisons are made within blocks. It is
generator-side and low-risk: a block is just another design column, and the existing deviation
(effect) coding in `build_model_matrix` already expands a categorical block column into contrast
columns, so **fitting a blocked design needs no analysis change** — the block enters the model
like any categorical factor.

```python
def randomized_complete_block(
    treatments: Sequence[Factor] | int, *, n_blocks: int, seed: int | None = None,
) -> Design: ...
def latin_square(treatments: int, *, seed: int | None = None) -> Design: ...
def blocked_factorial(
    factors: Sequence[Factor], *, block_generators: Sequence[str], seed: int | None = None,
) -> Design: ...
```

- **`randomized_complete_block`** — every treatment once per block; blocks randomized
  independently. The block is carried as a `CategoricalFactor`-style column (decision in §3.1).
- **`latin_square`** — a `k×k` arrangement with each treatment once per row and once per column
  (two blocking directions); rows/columns are the two nuisance factors. Validates squareness.
- **`blocked_factorial`** — assigns a `2^(k−p)` factorial to `2^q` blocks by confounding chosen
  interaction contrasts with blocks (`block_generators` like `"ABC"`), reusing the defining-
  relation machinery already in `fractional_factorial`. `meta` records which effects are
  confounded with blocks (they become inestimable — surfaced, not hidden).
- **Run-order utilities.** Extend `Design.randomize` with block-aware behaviour (randomize within
  block, keep blocks intact) — the same plot-aware machinery §2.1 adds, generalized so blocks and
  whole plots share one "randomize within groups" implementation.

### 3.1 Block representation decision (to settle in build)

Two options for carrying the block: (a) a dedicated `Design.blocks: tuple[int, ...] | None` field
symmetric to `whole_plots`, or (b) a reserved `"block"` `CategoricalFactor` in the `FactorSet`.
Option (b) makes the block fall out of the existing model-matrix expansion for free (no fit
change at all) but muddies "factor" vs "nuisance"; option (a) is cleaner conceptually but needs
`build_model_matrix` to know to add block contrast columns. **Lean (b)** for Phase 5c (least new
code, blocking is genuinely modeled as a categorical), and revisit if the semantics grate. Decide
before writing the generator; document whichever wins in `SERIALIZATION.md`.

---

## 4. Public API additions (`__init__.py`)

Generators: `definitive_screening`, `split_plot`, `randomized_complete_block`, `latin_square`,
`blocked_factorial`. Analysis: `fit_gls`. Factors/containers: the `hard_to_change` flag is a
constructor kwarg on the existing factor types (no new type). `Design.whole_plots` (and
`blocks`, per §3.1) are new fields, not new exports. Flat top-level namespace, as established.

---

## 5. Tests (anchors)

**`test_screening.py`**
- `definitive_screening` on `k = 4` continuous factors → exactly `2·4 + 1 = 9` runs, three
  levels `{−1, 0, +1}`, one all-zero center run, exactly two zeros per factor column.
- Main-effects orthogonality: `XᵀX` for the main-effect model is diagonal; main effects are
  orthogonal to all quadratic columns (the DSD defining property).
- `k = 6` reproduces the 13-run design tabulated in Jones & Nachtsheim (2011) up to row/sign
  conventions.
- Odd `k = 5` auto-adds one fake factor → `2·5 + 3 = 13` runs, fake column dropped.
- A quadratic fit on a DSD with an injected curved response recovers the active main effects and
  the nonzero quadratic term; >2-level categorical raises with a helpful message.

**`test_splitplot.py`**
- `split_plot` groups runs into the expected whole plots; hard-to-change factor is constant
  within each whole plot; `randomize(seed=0)` never splits a whole plot and is reproducible.
- `whole_plots` round-trips through `replicate` / `project` / `to_dict` / `from_dict`.
- REML on a textbook split-plot (Montgomery §14 or equivalent) recovers the published
  `σ²_wp` / `σ²` and the whole-plot & sub-plot standard errors to tolerance.
- **The trap:** `fit_ols` on the same split-plot data reports a *smaller* whole-plot standard
  error than `fit_gls` — asserted directly, so the regression that would reintroduce the
  anticonservative bug is caught.
- `fit_gls` on a design with `whole_plots is None` raises.

**`test_blocking.py`**
- `randomized_complete_block(4 treatments, n_blocks=3)` → every treatment once per block, 12
  runs; blocks randomized.
- `latin_square(5)` → 25 runs, each treatment once per row and once per column.
- `blocked_factorial` confounds the requested interaction with blocks and records it in `meta`;
  a fit shows the confounded effect is inestimable while the rest are clean.

Run gates unchanged: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.

---

## 6. Build order

**Phase 5a (independent, ship first — generator only, no analysis change):**
1. Conference-matrix construction + `definitive_screening` (continuous-only) + orthogonality /
   run-count / Jones–Nachtsheim anchors in `test_screening.py`.
2. Categorical DSD extension + its anchors.
3. Vignette: "Definitive screening designs" (six factors, one 13-run DSD, `half_normal_plot`
   → active few → reduced quadratic fit) — re-run `scripts/build_vignette_assets.py`.

**Phase 5b (container → generator → REML fit, each step green):**
4. `hard_to_change` flag + `Design.whole_plots` field + plot-aware `randomize` + serialization
   dispatch + carry-through tests.
5. `split_plot` generator (design generation only; fit not yet wired) + grouping/constancy
   anchors.
6. `variance.py` REML + `fit_gls` in `fit.py` + the variance-component and OLS-vs-GLS-trap
   anchors; `anova_table` two-stratum convention.
7. Vignette: "Split-plot designs for hard-to-change factors" (declare a hard-to-change factor →
   `split_plot` → `fit_gls` vs `fit_ols`, show the standard-error difference) — re-run
   `scripts/build_vignette_assets.py`.

**Phase 5c (generator-side, orthogonal):**
8. Block representation decision (§3.1) → `randomized_complete_block` / `latin_square` /
   `blocked_factorial` + block-aware run-order utility (shared with §2.1) + anchors.
9. Vignette (optional): "Blocking a factorial" — re-run assets. Then update `PLAN.md`'s roadmap
   line, `CLAUDE.md`'s status paragraph, and `SERIALIZATION.md`.

Test-first throughout; each step keeps pytest + ruff + mypy green.

---

## 7. Resolved decisions and deferred work

- **DSD analysis:** no new fit code — the value is the design; the existing OLS/RSM path consumes
  it, and the recommended reduce-then-fit strategy lives in the vignette, not in a function.
- **Two strata only:** split-split-plots (≥3 strata) and strip-plot / split-block designs are
  deferred — they multiply covariance structures for a niche payoff at this stage.
- **REML in-house:** a one-parameter REML over the variance ratio with block-diagonal `V⁻¹` is
  small and dependency-free; `statsmodels` stays an optional extra, not promoted to a hard dep.
- **Block as categorical (lean):** blocking is modeled as a reserved categorical column so the
  existing model-matrix expansion fits it for free (§3.1) — revisit only if the semantics grate.
- **`fit_ols` untouched:** `fit_gls` is a separate opt-in front door; no existing fit changes
  behaviour, so every Phase 1–4 result is bit-stable.
- **Mixed hard-to-change + mixture / space-filling:** out of scope — split-plot pairs with the
  factorial/RSM families first; combining it with the simplex or QMC generators is a later idea.

---

## 8. After Phase 5 (Phase 6 preview)

With Phase 5 done, the classical DoE surface — screening, RSM, optimal, space-filling, mixture,
split-plot, blocking — is essentially complete. The natural Phase 6 pool trends toward *richer
analysis and modern extensions* rather than new generator families:

- **Bayesian / model-averaged analysis** — posterior effect probabilities for supersaturated and
  DSD screens (where the frequentist "fit the active subset" step is a judgement call), and
  Bayesian D-/I-optimal design that averages over model uncertainty rather than fixing one model.
- **Generalized linear model analysis** — logistic / Poisson response fitting for binary and
  count experiments, reusing the coded model matrix but swapping OLS for IRLS.
- **Deferred items carried forward** — combined mixture-process designs, simplex-constrained
  optimization, and pseudo-components (from Phase 4); split-split-plots and strip-plots (from
  Phase 5).

That — plus whatever the accumulated real-world usage surfaces as missing — is the Phase 6 pool.
