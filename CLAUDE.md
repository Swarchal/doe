# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DoE (Design of Experiments) — a Python library for design-of-experiment analysis.

## Status

Phase 1 (factors/coding, the `Design` container, factorial generators, OLS analysis,
effect/Pareto/half-normal plots), Phase 2a (response-surface designs, quadratic fitting,
ANOVA + lack-of-fit, contour/diagnostic plots), and Phase 2b (surface optimization —
stationary point + canonical analysis, constrained optimum, Derringer–Suich desirability,
3-D `surface_plot`) are implemented. `plackett_burman` is still a deliberate
`NotImplementedError` stub, and categorical-factor model expansion is deferred
(`build_model_matrix` rejects categorical factors explicitly). See `docs/PLAN.md` for the
full roadmap and `docs/PHASE2.md` for the Phase 2 build plan.

## Commands

The project uses `uv` and a `src/` layout. Install and work in editable mode:

```bash
uv venv && uv pip install -e '.[dev]'   # create env + install with dev extras
uv run pytest                            # run the test suite
uv run pytest tests/test_fit.py::test_fit_recovers_known_effects   # a single test
uv run ruff check .                      # lint
uv run mypy                              # type-check (strict; checks src/)
```

## Architecture

Everything flows through a coded design matrix. Generation produces a `Design`; analysis
consumes one.

- `factors.py` — `ContinuousFactor`/`CategoricalFactor` and `FactorSet`. Owns the
  natural↔coded translation: continuous factors map to `[-1, +1]`. Designs are built and
  fitted in *coded* units but stored/reported in *natural* units.
- `design.py` — `Design`: a `pandas.DataFrame` of runs (natural units) plus its `FactorSet`.
  `.coded()` is the bridge to analysis; `.replicate()` and `.randomize()` handle replication
  and run order. The optional `point_types` tuple tags each run (e.g. `"center"`); it drives
  `n_center`/`center_indices`, which the lack-of-fit pure-error estimate depends on, and is
  carried through `replicate`/`randomize` so center labels survive.
- `generators/factorial.py` — `full_factorial` and `fractional_factorial` (2-level, from
  generator strings like `"D=ABC"`). `plackett_burman` is a `NotImplementedError` stub.
- `generators/rsm.py` — second-order designs: `central_composite` (factorial core + axial
  points + center replicates; `alpha` is `"faced"`/`"rotatable"`/`"orthogonal"` or a float)
  and `box_behnken` (3-level, no corner runs). Both require continuous factors and set
  `point_types`.
- `analysis/model.py` — `build_model_matrix` expands a `Design` into intercept + main-effect
  + interaction (+ optional quadratic) columns in coded units. Squared terms are only emitted
  for factors that actually take values off `{-1, +1}` (a pure ±1 column squares to the
  intercept). Categorical factors are rejected here (contrast expansion deferred).
- `analysis/fit.py` — `fit_ols` returns a `FitResult` (coefficients, effects, std errors,
  t/p-values, `conf_int`, `r_squared`). Note: in coded units an *effect* is `2 × coefficient`
  (the −1→+1 swing), which the tests rely on. A saturated model (residual dof 0) warns and
  yields NaN standard errors rather than erroring.
- `analysis/anova.py` — `anova_table` (sequential/Type I SS via QR), `lack_of_fit` (needs ≥2
  center points for pure error), and predictive metrics `press`/`predicted_r2`/`adjusted_r2`.
- `analysis/optimize.py` (Phase 2b) — reads the fitted quadratic as `ŷ = b₀ + xᵀb + xᵀB x`
  (`_quadratic_form` pulls `b`/symmetric `B` from `term_names`). `stationary_point` solves
  `−½ B⁻¹ b` + canonical eigen-analysis (max/min/saddle); `optimum` does a constrained multistart
  `L-BFGS-B` search over the coded box (reports `at_bound`); `desirability` maximizes a
  Derringer–Suich geometric-mean `D` over `ResponseGoal`s via `differential_evolution`.
- `plotting.py` — effect plots (`pareto_plot`, `main_effects_plot`, `half_normal_plot`),
  RSM (`contour_plot` + 3-D `surface_plot` over their headless core `surface_grid`, which takes
  `fixed={factor: value}` to slice >2 factors), and diagnostics (`residuals_vs_fitted`,
  `normal_qq`). Imports `matplotlib` lazily so the core library stays usable without the optional
  `plotting` extra.

Tests anchor correctness against known designs/effects (e.g. a 2^(4-1) defining relation,
recovering injected coded-unit effects exactly, textbook CCD/Box-Behnken run counts). Keep
that pattern when extending.

## Docs

`docs/VIGNETTES.md` is a worked, narrative tour of the library. Every console output and
figure in it is *real*: `scripts/build_vignette_assets.py` runs each example, writes figures
to `docs/img/`, and prints the outputs for transcription. When you change behaviour that a
vignette demonstrates, re-run `uv run python scripts/build_vignette_assets.py` and update the
transcribed numbers/figures so the doc stays truthful.

## Code

Use modern python, with typing, pyproject.toml and uv for packaging.
Libraries used should be standard sci-py libraries. So pandas, numpy, matplotlib,
scipy etc.
Code should be accompanied by unit tests.
