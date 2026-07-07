# UX & API Ergonomics Plan

Detailed plan for a round of **user-experience improvements** to the existing library. Unlike
`PHASE2`–`PHASE4`, this adds **no new DoE capability** — no new design families, criteria,
models, or plots. Every change here targets the *seams* of the current API: how a user
carries state from one call to the next, how objects present themselves in a notebook, and
how discoverable the workflow is. The internal machinery (generators → `Design.coded()` →
model matrix → OLS → diagnostics) is well-factored and stays exactly as it is; this work sits
on top of it.

> **Scope decision:** This is a *polish* pass, explicitly bounded to ergonomics. If a change
> would add a new capability, it belongs in a phase doc, not here. The test is: *could the user
> already do this, just more awkwardly?* If yes, it is in scope. If it lets them do something
> new, it is out.

> **Compatibility decision:** Every change is **additive and backward-compatible**. The
> existing positional/array-based paths and free-function analysis API keep working unchanged
> and remain the tested core. New conveniences are thin delegators over them, never
> replacements. No import in `docs/VIGNETTES.md` should break; the vignette is *updated* to
> show the nicer path, not rewritten because the old one broke.

## Motivation

The library is well-factored internally, so the friction is all at the boundaries between
calls — the places where the user has to manually thread state that the objects could carry
for them. Five concrete pain points, in rough order of felt impact:

1. **Responses are detached, positionally-aligned numpy arrays.** `fit_ols(design, gfp)` takes
   a bare array that the caller must keep row-aligned with `design.runs` by hand. The vignette
   has to actively warn about this (`replicate(3, each=True)` is recommended because it
   "can't misalign"). Misalignment is silent and produces a plausible-but-wrong fit.

2. **Analysis free-functions re-thread `(result, design, response)`.** `anova_table`,
   `lack_of_fit` all demand the same three arguments the user already passed to `fit_ols`.
   `FitResult` *already* forwards `.stationary_point()` / `.optimum()` — but stops there, so
   the API is half-fluent and inconsistent: some post-fit analysis hangs off the result, most
   does not.

3. **No notebook reprs.** DoE work happens in notebooks, yet `Design`, `FactorSet`, and
   `FitResult` all fall back to the default dataclass repr — typing `design` shows a struct
   dump, not the runs table. `interactive.to_html` already knows how to render a `Design`
   beautifully; nothing wires it to `_repr_html_`.

4. **`summary()` returns a `dict[str, tuple[float, float]]`.** The natural artifact of a fit is
   a coefficient table (coef, effect, std error, t, p, CI). Every stats package returns a
   table; returning a dict of 2-tuples forces the user to build the table themselves.

5. **No `predict()`.** After fitting, scoring a new point in natural units means hand-building a
   model-matrix row. The information to do it (factors, coding, term names, coefficients) is
   all on the `FitResult`, but there is no method to use it.

None of these are DoE-capability gaps — they are ergonomics.

## Goals

By the end of this work a user can:

1. **Attach a response to a design as data**, so it travels with the runs and cannot silently
   misalign: `design.with_response("gfp", values)`, then `fit_ols(design, "gfp")`.
2. **Do all post-fit analysis off the `FitResult`** without re-passing the design and response:
   `result.anova()`, `result.lack_of_fit()`, `result.press()`, `result.predicted_r2()`,
   `result.adjusted_r2()`, `result.predict(new_points)` — finishing the fluent pattern that
   `.stationary_point()` / `.optimum()` already started.
3. **See a `Design`, `FactorSet`, or `FitResult` render meaningfully** in a notebook and at the
   REPL — a runs table / factor summary / coefficient table, not a dataclass dump.
4. **Get a coefficient table as a DataFrame** from `summary()`, ready to print, filter, or
   export.
5. Rely on **consistent generator signatures** — pass a built `FactorSet` anywhere a list of
   factors is accepted, and find keyword-only options everywhere they are keyword-only
   *somewhere*.

Compatibility anchors (the existing tests must keep passing unchanged): `fit_ols(design,
array)` still fits; `anova_table(result, design, response)` still returns the same table;
`summary()`'s data is still reachable; every generator still accepts a plain `list[Factor]`.

## Work breakdown

Four independently-shippable, PR-sized phases. A and B carry ~80% of the felt improvement; if
only one slice ships, ship those.

### Phase A — Response as first-class data

The highest-leverage change. Make the response a column that lives on the `Design`, so it is
validated once and carried through every transformation.

- **`Design.with_response(name: str, values) -> Design`** — returns a *new* `Design` with a
  length-checked response column appended to `runs`. Rejects a length mismatch loudly (the
  failure that is currently silent). Does not mutate the original (Designs are treated as
  values elsewhere — `replicate`/`randomize` already return new objects).
- **`replicate` / `randomize` carry response columns.** Any non-factor column already rides
  along in `runs`, but confirm and test that responses survive replication (with `each=True`
  producing the correct row multiplicity) and randomization (staying aligned to their runs).
- **`fit_ols(design, response=...)` accepts a column name** *or* an array (today's behavior).
  A `str` is looked up in `design.runs`; anything array-like is used as-is. One extra branch;
  the array path is untouched and stays the tested core.
- **Vignette update:** switch the loading-efficiency example to
  `design.with_response("gfp", gfp)` → `fit_ols(rep, "gfp")`, and delete the misalignment
  caveat that the new path makes structurally impossible. Re-run
  `scripts/build_vignette_assets.py` so transcribed numbers stay truthful.

*Anchors:* `with_response` round-trips values through `replicate`/`randomize`; a length
mismatch raises `ValueError`; `fit_ols` gives identical coefficients whether handed the column
name or the raw array.

### Phase B — Fluent post-fit analysis

Finish the pattern `FitResult` already began, so the result object is a sufficient handle for
everything downstream.

- **`fit_ols` stashes what it used.** Store the originating `design` and the `response` array on
  the `FitResult` (new fields, defaulted so direct construction in tests still works).
- **Delegating methods on `FitResult`:** `.anova()`, `.lack_of_fit()`, `.press()`,
  `.predicted_r2()`, `.adjusted_r2()` — each a one-liner forwarding to the existing free
  function with the stashed design/response. The free functions remain the implementation and
  keep their current signatures and tests; these methods are sugar.
- **`FitResult.predict(points) -> np.ndarray`** — accept new runs in *natural* units (a dict,
  a `DataFrame`, or a `Design`), code them through the stored `FactorSet`, expand with the
  same term structure (`order`/`interactions`/model path) via the existing
  `expand_coded_points`, and return `X_new @ coefficients`. This is the one genuinely new
  affordance, but it exposes an ability the object already has all the pieces for — not a new
  DoE method.

*Anchors:* `result.anova()` equals `anova_table(result, design, response)` cell-for-cell;
`result.predict(design_rows)` reproduces `result.fitted` on the training points; `predict`
honors the mixture/Scheffé path when the fit was a blending model.

### Phase C — Display & notebook polish

Make the objects present themselves. DoE is a notebook-first workflow; the reprs are the UX.

- **`Design._repr_html_`** — reuse `interactive.to_html` (already renders a Design as a
  sortable natural+coded run sheet). **`Design.__repr__`** — a compact one-liner:
  `Design('loading', 12 runs × 2 factors, 4 center)`.
- **`FitResult.summary()` returns a `pandas.DataFrame`** (index = term names; columns = coef,
  effect, std_error, t, p, and CI bounds). Keep the old mapping reachable as
  `summary_dict()` for anyone depending on it (soft, non-breaking). **`FitResult._repr_html_` /
  `__repr__`** show that table plus `R²` / adjusted `R²`.
- **`FactorSet.__repr__` and factor `__repr__`s** — list names, natural ranges, and units, so
  inspecting factors at the REPL is legible.

*Anchors:* reprs are display-only and must not change any computed value; `summary()` DataFrame
carries the same numbers the dict did; `to_html` is unchanged and simply reused.

### Phase D — Signature consistency (low-risk cleanups)

Small correctness-neutral smoothing so the API feels uniform.

- **Accept `FactorSet | Sequence[Factor]` in every generator.** They already re-wrap a list into
  a `FactorSet` internally (`fs = FactorSet(factors)`); make that wrap a no-op when handed an
  existing `FactorSet`, so a built factor set round-trips instead of erroring.
- **`full_factorial`'s `levels` becomes keyword-only**, matching every other generator's
  keyword-only style (`central_composite`, `box_behnken`, `latin_hypercube`, … all use `*`).
  This is the one place a positional option leaks; call sites in tests/vignette pass it by
  keyword already, so the blast radius is small — but verify.
- **Document `model=` as the front door** over `order=`/`interactions=` in the `fit_ols`
  docstring (keep both; `model=` is already the clearer, self-describing spelling the vignette
  uses).

*Anchors:* every existing generator call still type-checks and runs; a `FactorSet` and the
equivalent `list` produce identical designs.

## Sequencing & risk

- **A and B are the payload** and are independent of each other; either can ship first. A
  removes the silent-misalignment footgun; B removes the repeated-argument tax.
- **C is orthogonal** and can land any time — pure display, zero computational surface.
- **D is a mop-up** best done last so it doesn't churn signatures other phases touch.
- Total surface: ~2 new methods on `Design`, ~6 on `FitResult`, three `_repr_html_`/`__repr__`
  pairs, one `fit_ols` branch, one signature tweak. No module added, no dependency added, no
  behavior removed.

## Explicit non-goals

- No new design generators, optimality criteria, model forms, or plots — that is phase work.
- No breaking changes: the array/free-function API stays and stays tested; new methods delegate
  to it.
- No mutation-based/"builder" API on `Design` — Designs stay value objects that return new
  Designs, consistent with `replicate`/`randomize`.
- No new heavy dependencies (e.g. a rich-repr or tabulation library); reprs use pandas/HTML the
  library already ships.
