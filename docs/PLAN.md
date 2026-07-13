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
- Definitive screening designs (Jones–Nachtsheim conference matrices)

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

## 5. Interchange & delivery

- Serialization: a versioned `Design.to_dict`/`from_dict` round-trip plus schema validation, so a
  design can be handed to another tool (e.g. a liquid-handler protocol generator) and back
- HTML run sheets for taking a design to the bench
- An HTTP API over the library, for callers that aren't Python

## Proposed package layout

This is the layout as built:

```
src/doe/
  factors.py          # Continuous/Categorical/Mixture factors, FactorSet, coding/scaling
  design.py           # Design container (replicate/randomize/project, point types, to_dict)
  serialization.py    # versioned schema + validate_design_dict for the Design payload
  interactive.py      # to_html: self-contained sortable run sheet
  generators/
    factorial.py      # full, fractional, Plackett-Burman
    rsm.py            # CCD, Box-Behnken
    screening.py      # definitive screening designs
    optimal.py        # D/I-optimal coordinate exchange, candidate_grid, augment
    spacefilling.py   # LHS, Sobol, Halton
    mixture.py        # simplex-lattice/centroid, extreme vertices, mixture candidates
  analysis/
    model.py          # term list -> model matrix (incl. Scheffé blending models)
    fit.py            # OLS, effects, confidence/prediction intervals
    anova.py          # ANOVA table, lack-of-fit, PRESS/predicted R²
    diagnostics.py    # efficiency, VIF, leverage, alias/correlation, coverage metrics
    optimize.py       # stationary point, constrained optimum, desirability
  plotting.py
tests/                # mirrors the above; validate against known textbook designs
```

A `doe-service/` workspace member wraps the library in a stateless FastAPI HTTP API
(`doe_service` imports `doe`, never the reverse, so `doe` stays scipy-stack-only). See
[`WEBSERVICE.md`](WEBSERVICE.md), [`WEBSERVICE_API.md`](WEBSERVICE_API.md) and
[`WEBSERVICE_BUILD.md`](WEBSERVICE_BUILD.md).

## Dependencies

Required: `numpy`, `scipy`, `pandas`. Optional extras: `plotting` (`matplotlib` — imported
lazily, so the core library works without it), `stats` (`statsmodels`), `docs` (Sphinx) and
`dev` (`pytest`, `ruff`, `mypy`). Regression/ANOVA is implemented directly on the scipy stack
rather than delegated to `statsmodels`. Packaging is `pyproject.toml` + `uv`, as a workspace
whose root member is `doe` and whose other member is `doe-service`.

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
4. **Phase 4 — Specialized** *(done)*: space-filling (LHS/QMC) and mixture designs.
   (Multi-response desirability optimization, originally slated here, shipped early in Phase 2b.)
   **Phase 4a** (space-filling: `latin_hypercube`/`sobol`/`halton` +
   `discrepancy`/`maximin_distance`) and **Phase 4b** (mixture designs: `MixtureFactor`,
   `simplex_lattice`/`simplex_centroid`/`extreme_vertices`/`mixture_candidates`, Scheffé
   blending models, `ternary_contour`) are both *done*. See [`PHASE4.md`](PHASE4.md) for the
   detailed build plan.
5. **Phase 5 — Screening & restricted randomization** *(in progress)*. See
   [`PHASE5.md`](PHASE5.md) for the detailed build plan.
   - **Phase 5a — Definitive screening designs (DSD)** *(done)*: `definitive_screening` builds
     Jones–Nachtsheim conference-matrix designs that screen main effects and detect
     curvature/2FIs in few runs, avoiding the full-factorial → CCD two-stage flow. Reuses the
     existing OLS/RSM analysis machinery unchanged.
   - **Phase 5b — Split-plot / hard-to-change factors** *(not yet started)*: restricted
     randomization for factors that cannot be reset every run (the industrial norm). Touches the
     analysis layer, not just generation: needs a whole-plot/sub-plot structure on `Design` and a
     GLS/REML fit path rather than OLS.
   - **Phase 5c — Classical / blocking** *(not yet started)*: randomized complete block, Latin
     square, blocked factorials (fractions assigned to blocks via defining contrasts), richer
     run-order utilities.

Alongside the phases, serialization (`Design.to_dict`/`from_dict` + schema validation), the HTML
run sheet, and the `doe-service` HTTP API (v1 complete) have all shipped.

A good correctness anchor throughout: reproduce canonical designs from Montgomery's *Design and
Analysis of Experiments* in tests, so generated designs are verified against published references.
