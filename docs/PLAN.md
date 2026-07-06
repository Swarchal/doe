# DoE Library Plan

A plan for what a fully-featured Design of Experiments (DoE) library could contain,
organized by capability area, with a proposed package layout and a phased roadmap so it
can be built incrementally.

## Scope: the DoE problem

A DoE library has two halves that meet in the middle at a **design matrix**:

1. **Design generation** — given factors and constraints, produce the set of experimental runs.
2. **Analysis** — given a design plus measured responses, fit a model and tell the user what matters.

Everything below hangs off those two halves.

## 1. Core abstractions (the foundation everything depends on)

- **`Factor`** types: continuous (with low/high bounds), categorical (discrete levels), ordinal.
  Carry name, units, and natural-vs-coded representation.
- **Coding / scaling**: convert between natural units and coded units (continuous → `[-1, +1]`,
  categorical → contrast/dummy encoding). This is pervasive — designs are generated in coded
  space, results reported in natural space.
- **`Design`** object: a `pandas.DataFrame` of runs wrapped with metadata (factor definitions,
  run order, blocks, center points, coding info). The shared currency between generation and analysis.
- **Design diagnostics**: condition number, VIF, correlation/alias matrix, D-/A-/G-efficiency,
  leverage — used to *evaluate* any design regardless of how it was made.

## 2. Design generators

**Screening / factorial**
- Full factorial (2-level and general mixed-level)
- Fractional factorial `2^(k-p)` with generators, defining relation, resolution, and alias structure
- Plackett-Burman (efficient main-effects screening)

**Response surface methodology (RSM)**
- Central composite designs (circumscribed/inscribed/face-centered)
- Box-Behnken designs

**Optimal (computer-generated) designs**
- D-, A-, I-, G-optimality criteria
- Candidate-set generation + exchange algorithms (Fedorov, coordinate-exchange) — handles irregular
  constraints, custom models, and odd run-count budgets

**Space-filling** (for simulation / computer experiments)
- Latin Hypercube Sampling (maximin, correlation-minimizing)
- Sobol / Halton sequences — can lean on `scipy.stats.qmc`

**Mixture designs**
- Simplex-lattice, simplex-centroid, extreme-vertices (constrained)

**Classical/blocking**
- Randomized complete block, Latin square, randomization & run-order utilities

## 3. Analysis

- Model specification (main effects, interactions, quadratic terms; Wilkinson-style formula or
  explicit term list)
- Fitting via OLS (`numpy.linalg`/`scipy`, or optionally `statsmodels`)
- Effect estimates, coefficients, confidence intervals
- ANOVA table, lack-of-fit test, R²/adjusted R²/Q²
- RSM optimization: stationary-point analysis, canonical analysis, numerical optimization of fitted
  surface, multi-response desirability functions

## 4. Visualization (`matplotlib`)

- Main-effect and interaction plots
- Pareto chart of effects; half-normal / normal probability plot of effects
- Residual diagnostics (residuals vs fitted, normal Q-Q)
- RSM contour and 3D surface plots
- Correlation/alias heatmaps

## Proposed package layout

```
src/doe/
  factors.py          # Factor, FactorSet, coding/scaling
  design.py           # Design container + diagnostics
  generators/
    factorial.py      # full, fractional, Plackett-Burman
    rsm.py            # CCD, Box-Behnken
    optimal.py        # D/A/I-optimal, exchange algorithms
    spacefilling.py   # LHS, Sobol, Halton
    mixture.py
  analysis/
    model.py          # formula/terms, design->model matrix
    fit.py            # OLS, effects, ANOVA
    optimize.py       # RSM optimization, desirability
  plotting.py
tests/                # mirrors the above; validate against known textbook designs
```

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`. Consider `statsmodels` (optional extra) for richer
ANOVA/regression rather than reimplementing it. `pyproject.toml` + `uv`, with `pytest` for tests
and an `[optional]` group for plotting/statsmodels.

## Phased roadmap

1. **Phase 1 — Foundation** *(done)*: `Factor`, coding, `Design` container, full + fractional
   factorial, basic OLS fit + effects, main-effect/Pareto plots. This alone is a usable
   screening library.
2. **Phase 2 — RSM** *(done)*: CCD, Box-Behnken, quadratic model fitting, ANOVA, contour/surface
   plots, plus surface optimization + desirability. See [`PHASE2.md`](PHASE2.md) for the detailed
   build plan.
3. **Phase 3 — Optimal designs** *(done)*: coordinate-exchange engine + D/I-optimality, design
   diagnostics (efficiency, VIF, alias matrix, leverage). See [`PHASE3.md`](PHASE3.md) for the
   detailed build plan.
4. **Phase 4 — Specialized:** space-filling (LHS/QMC) and mixture designs. (Multi-response
   desirability optimization, originally slated here, shipped early in Phase 2b.) **Phase 4a**
   (space-filling: `latin_hypercube`/`sobol`/`halton` + `discrepancy`/`maximin_distance`) is
   *done*; **Phase 4b** (mixture designs) is *not yet started*. See [`PHASE4.md`](PHASE4.md) for
   the detailed build plan.

A good correctness anchor throughout: reproduce canonical designs from Montgomery's *Design and
Analysis of Experiments* in tests, so generated designs are verified against published references.
