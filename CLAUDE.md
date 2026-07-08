# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DoE (Design of Experiments) — a Python library for design-of-experiment analysis.

## Status

Phase 1 (factors/coding, the `Design` container, factorial generators, OLS analysis,
effect/Pareto/half-normal plots), Phase 2a (response-surface designs, quadratic fitting,
ANOVA + lack-of-fit, contour/diagnostic plots), and Phase 2b (surface optimization —
stationary point + canonical analysis, constrained optimum, Derringer–Suich desirability,
3-D `surface_plot`) are implemented, completing Phase 1 and Phase 2. Phase 3
(computer-generated optimal designs — `coordinate_exchange` engine with D/I-optimality,
`candidate_grid`, `augment` — plus design diagnostics: `efficiency` (D/A/G/I),
`vif`, `condition_number`, `correlation_matrix`/`alias_matrix`, `leverage`) is implemented.
Phase 4a (space-filling generators — `latin_hypercube`, `sobol`, `halton`, thin wrappers over
`scipy.stats.qmc` — plus the model-free coverage diagnostics `discrepancy` and
`maximin_distance`) is implemented. Phase 4b (mixture designs — the `MixtureFactor` proportion
type, `simplex_lattice`/`simplex_centroid`/`extreme_vertices` generators plus
`mixture_candidates` feeding the Phase 3 optimal engine, Scheffé no-intercept blending models in
`build_model_matrix`/`fit_ols`, and the `ternary_contour` plot) is implemented, completing
Phase 4. Phase 5 (screening & restricted randomization — definitive screening designs,
split-plot/hard-to-change factors, classical/blocking) is not yet started; see
`docs/PLAN.md`/`docs/PHASE4.md` §7 for the scoped pool. `build_model_matrix` expands categorical
factors via deviation (effect) coding, so OLS analysis handles mixed continuous/categorical
designs. See `docs/PLAN.md` for the full roadmap and
`docs/PHASE2.md`/`docs/PHASE3.md`/`docs/PHASE4.md` for the detailed build plans.

## Commands

The project uses `uv` and a `src/` layout. Install and work in editable mode:

```bash
uv venv && uv pip install -e '.[dev]'   # create env + install with dev extras
uv run pytest                            # run the test suite
uv run pytest tests/test_fit.py::test_fit_recovers_known_effects   # a single test
uv run ruff check .                      # lint
uv run mypy                              # type-check (strict; checks src/)
uv run --extra docs sphinx-build -b html docs docs/_build/html   # build the HTML docs
```

CI (`.github/workflows/ci.yml`) runs these same three checks — `ruff check .`, `mypy`,
and `pytest` — via `uv run --extra dev` on a Python 3.11/3.12/3.13 matrix, on pushes to
`main` and on pull requests.

## Architecture

Everything flows through a coded design matrix. Generation produces a `Design`; analysis
consumes one.

- `factors.py` — `ContinuousFactor`/`CategoricalFactor`/`MixtureFactor` and `FactorSet`. Owns
  the natural↔coded translation: continuous factors map to `[-1, +1]`. Designs are built and
  fitted in *coded* units but stored/reported in *natural* units. `MixtureFactor` (Phase 4b) is
  a proportion in `[low, high] ⊆ [0, 1]`; its columns are *not* rescaled (`coded()` passes them
  through as proportions — the simplex has no box coding). A `FactorSet` is all-mixture or
  mixture-free (`is_mixture` property; mixing raises), and mixture bounds must leave a feasible
  blend (`Σ low ≤ 1 ≤ Σ high`).
- `design.py` — `Design`: a `pandas.DataFrame` of runs (natural units) plus its `FactorSet`.
  `.coded()` is the bridge to analysis; `.replicate()` and `.randomize()` handle replication
  and run order; `.project(names)` narrows the design to a subset of its factors (dropping the
  other factors' columns while runs/responses/`point_types` ride along) — the "project onto the
  survivors" step after screening, feeding the projected design to `augment`. The optional
  `point_types` tuple tags each run (e.g. `"center"`); it drives `n_center`/`center_indices`,
  which the lack-of-fit pure-error estimate depends on, and is carried through
  `replicate`/`randomize`/`project` so center labels survive.
- `generators/factorial.py` — `full_factorial` and `fractional_factorial` (2-level, from
  generator strings like `"D=ABC"`), plus `plackett_burman` (saturated, orthogonal
  main-effect screening designs; run counts from Sylvester doubling of the order-1/12/20
  base Hadamard matrices, picking the next available size when one isn't constructible).
- `generators/rsm.py` — second-order designs: `central_composite` (factorial core + axial
  points + center replicates; `alpha` is `"faced"`/`"rotatable"`/`"orthogonal"` or a float)
  and `box_behnken` (3-level, no corner runs). Both require continuous factors and set
  `point_types`.
- `generators/optimal.py` (Phase 3b) — computer-generated designs via a `coordinate_exchange`
  engine (Meyer–Nachtsheim): seeds a random feasible start over a coded `candidate_grid` (or a
  custom `region`), then exchanges each mutable run against every candidate to maximise
  `criterion="D"` (`log_det_information`) or minimise `"I"` (average prediction variance),
  restarting `n_restarts` times to dodge local optima. `d_optimal`/`i_optimal` are
  intention-revealing wrappers; `augment` holds an existing design's rows fixed
  (`point_type="existing"`) and searches only the added rows. Returns a plain `Design` with the
  search diagnostics stashed in `meta`.
- `generators/spacefilling.py` (Phase 4a) — space-filling designs that target *coverage* rather
  than model efficiency (computer experiments, surrogate modelling, exploratory sampling):
  `latin_hypercube` (stratified, `criterion="maximin"`/`"correlation"`/`None`), `sobol`
  (power-of-two run counts only — raises otherwise, naming the nearest valid sizes), and
  `halton` (any count). Thin wrappers over `scipy.stats.qmc`: samples in `[0, 1]^k` map to
  coded `[-1, +1]` then decode to natural units. Continuous factors only; reproducible and
  self-describing via `meta` (`sampler`/`seed`/`criterion`/`scramble`).
- `generators/mixture.py` (Phase 4b) — mixture (simplex) designs whose rows sum to 1:
  `simplex_lattice` (`{k, m}` lattice), `simplex_centroid` (`2^k − 1` subset centroids),
  `extreme_vertices` (McLean–Anderson XVERT vertices of a bound-constrained simplex, + centroid),
  and `mixture_candidates` (a discrete simplex candidate set shaped like `candidate_grid` output,
  so it feeds the Phase 3 `coordinate_exchange` engine — the engine exchanges whole rows, so
  sum-to-1 is preserved; `d_optimal(..., region=mixture_candidates(...))` gives odd-budget
  D-optimal blends). All-mixture factor sets only; `point_types` tags vertices/centroids.
- `analysis/model.py` — `build_model_matrix` expands a `Design` into intercept + main-effect
  + interaction (+ optional quadratic) columns in coded units. Squared terms are only emitted
  for continuous factors that actually take values off `{-1, +1}` (a pure ±1 column squares to
  the intercept). Categorical factors are expanded by deviation (effect) coding into `k-1`
  contrast columns named `factor[level]` (first level is the `-1` reference); interactions are
  products of the participating factors' encoded columns. An all-mixture design instead takes
  the Scheffé path (`_scheffe_matrix`): no intercept, `order=1` linear blending (`Σ βᵢxᵢ`),
  `order=2` adds the `i<j` cross products (`interactions` is ignored). Term names are the
  component names and `A:B` products, so ANOVA/VIF/plots key off them unchanged.
- `analysis/fit.py` — `fit_ols` returns a `FitResult` (coefficients, effects, std errors,
  t/p-values, `conf_int`, `r_squared`). Note: in coded units an *effect* is `2 × coefficient`
  (the −1→+1 swing), which the tests rely on. A saturated model (residual dof 0) warns and
  yields NaN standard errors rather than erroring. Mixture designs fit Scheffé blending models
  (`model="scheffe-linear"`/`"scheffe-quadratic"`, which additionally *require* a mixture design):
  no intercept, R² stays *centered* (valid — the constant is in the Scheffé column space, so this
  avoids the inflated uncorrected-R² gotcha), and `effects` is all-NaN (the ±1 swing is
  meaningless on proportions).
- `analysis/anova.py` — `anova_table` (sequential/Type I SS via QR), `lack_of_fit` (needs ≥2
  center points for pure error), and predictive metrics `press`/`predicted_r2`/`adjusted_r2`.
  For a Scheffé (no-intercept) fit the table reports the `k` component columns as one
  `Linear blending` row with `k−1` df (their sequential SS minus the mean correction), then a
  1-df row per cross product — the textbook mixture-ANOVA convention.
- `analysis/diagnostics.py` — model-based design diagnostics (Phase 3a) that judge *any* design
  against a model: `information_matrix`/`condition_number`/`log_det_information`, `vif`,
  `leverage`, `correlation_matrix` (alias structure), and `efficiency` (D/A/G/I, normalised so
  an orthogonal design scores ~1, integrating scaled prediction variance over a candidate
  region). Plus model-free coverage metrics (Phase 4a): `discrepancy` (`qmc.discrepancy`) and
  `maximin_distance`, both on `Design.coded()` rescaled to the `[0, 1]^k` unit cube. Headless;
  the cores feed both the plotting wrappers and the `optimal.py` engine (whose D-objective *is*
  `log_det_information`).
- `analysis/optimize.py` (Phase 2b) — reads the fitted quadratic as `ŷ = b₀ + xᵀb + xᵀB x`
  (`_quadratic_form` pulls `b`/symmetric `B` from `term_names`). `stationary_point` solves
  `−½ B⁻¹ b` + canonical eigen-analysis (max/min/saddle); `optimum` does a constrained multistart
  `L-BFGS-B` search over the coded box (reports `at_bound`); `desirability` maximizes a
  Derringer–Suich geometric-mean `D` over `ResponseGoal`s via `differential_evolution`.
- `plotting.py` — effect plots (`pareto_plot`, `main_effects_plot`, `half_normal_plot`,
  `interaction_plot` over its headless `interaction_lines`), RSM (`contour_plot` + 3-D
  `surface_plot` over their headless core `surface_grid`, which takes `fixed={factor: value}` to
  slice >2 factors), model diagnostics (`residuals_vs_fitted`, `normal_qq`,
  `predicted_vs_actual`, `leverage_plot`), alias structure (`correlation_heatmap` over its
  headless `alias_matrix`), and mixtures (`ternary_contour` over its headless `ternary_grid` —
  a 3-component Scheffé blending surface on the simplex). Imports `matplotlib` lazily so the core
  library stays usable without the optional `plotting` extra.
- `serialization.py` — `validate_design_dict` checks a serialized-`Design` mapping (the
  `Design.to_dict` payload consumed by `Design.from_dict`) against the versioned schema, raising
  `ValidationError`; see `docs/SERIALIZATION.md`.
- `interactive.py` — `to_html` renders a `Design` to a self-contained, sortable HTML run sheet
  (natural + coded columns on a diverging colour scale, tinted point-type column); adds no new
  Python dependencies.

Tests anchor correctness against known designs/effects (e.g. a 2^(4-1) defining relation,
recovering injected coded-unit effects exactly, textbook CCD/Box-Behnken run counts). Keep
that pattern when extending.

## Docs

`docs/VIGNETTES.md` is a worked, narrative tour of the library. Every console output and
figure in it is *real*: `scripts/build_vignette_assets.py` runs each example, writes figures
to `docs/img/`, and prints the outputs for transcription. When you change behaviour that a
vignette demonstrates, re-run `uv run python scripts/build_vignette_assets.py` and update the
transcribed numbers/figures so the doc stays truthful.

`docs/WORKFLOW.md` is the shorter end-to-end walkthrough (factors → design → fit → optimum);
its outputs and `wf_*.png` figures come from `scripts/build_workflow_assets.py` the same way —
re-run it and update the transcriptions when the walkthrough's behaviour changes.

`docs/WORKFLOW2.md` is a companion walkthrough on *balancing two competing readouts* (yield vs
impurity → two OLS fits → Derringer–Suich `desirability`); its outputs and `wf2_*.png` figures
come from `scripts/build_workflow2_assets.py` the same way — re-run it and update the
transcriptions when its behaviour changes.

`docs/WORKFLOW3.md` is the *prequel* walkthrough on screening (six candidate factors →
16-run `fractional_factorial` screen → `half_normal_plot` picks the vital three → project the
screen onto the survivors and `augment` it into a quadratic-capable design → same operating
point as WORKFLOW.md for 24 runs total); its outputs and `wf3_*.png` figures come from
`scripts/build_workflow3_assets.py` the same way — re-run it and update the transcriptions
when its behaviour changes.

`docs/` is also a Sphinx project (`conf.py`, furo theme, MyST so the markdown guides build
as-is, napoleon for the Google-style docstrings). `docs/api/` holds one `automodule` page per
module; add a page there when adding a module. Build with
`uv run --extra docs sphinx-build -b html docs docs/_build/html` — keep it warning-free:
`.github/workflows/docs.yml` builds with `-W` on every push/PR and deploys the site to
GitHub Pages on pushes to `main`.

## Code

Use modern python, with typing, pyproject.toml and uv for packaging.
Libraries used should be standard sci-py libraries. So pandas, numpy, matplotlib,
scipy etc.
Code should be accompanied by unit tests.
