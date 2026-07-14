# Web service API specification (v1 draft)

The concrete API design for the stateless compute service planned in
[WEBSERVICE.md](WEBSERVICE.md) — the same relationship the PHASE docs have to PLAN.md.
Every endpoint below maps onto an existing public function; request fields mirror that
function's keyword arguments, so the OpenAPI schema doubles as library documentation.
The implementation build plan — milestones, library prep, stub map — is
[WEBSERVICE_BUILD.md](WEBSERVICE_BUILD.md). For a runnable `curl` example of every
endpoint (real captured request/response pairs), see the cookbook,
[WEBSERVICE_EXAMPLES.md](WEBSERVICE_EXAMPLES.md).

## Conventions

- **Base path `/v1`.** All compute endpoints are `POST` with a JSON body — these are
  operations, not resources, and design documents are too large for query strings.
  `GET /v1/health` and the auto-generated `/openapi.json` are the only non-POST routes.
- **The design document is the wire format.** Wherever a request has a `design` field
  it takes the `Design.to_dict()` document ([SERIALIZATION.md](SERIALIZATION.md));
  wherever a response returns one it came from `Design.to_dict()`. The service runs
  `validate_design_dict` on every inbound design after Pydantic shape-checking.
- **Natural units on the wire.** Runs, bounds, `fixed` values, and prediction points
  are natural-unit values, matching the serialization doctrine ("natural values are the
  public contract"). The two exceptions are explicit `region` candidate arrays and the
  `coded` echo in optimization results, which are coded by definition.
- **Responses are named columns.** Analysis endpoints take `response` as the *name* of
  a column on the design's runs — the `with_response` pairing that cannot silently
  misalign. Clients attach readouts via `POST /v1/designs/responses` (or by writing the
  column into the document themselves; the validator accepts extra run columns).
- **Model specification.** A single `model` field, accepted everywhere a model is
  needed, in either form the library uses:
  - a string: `"linear"`, `"quadratic"`, `"scheffe-linear"`, `"scheffe-quadratic"`;
  - an object: `{"order": 1 | 2, "interactions": true | false}`.

  Defaults match the library (`fit`: linear; `optimal` generators: quadratic;
  `diagnostics`: `{"order": 1, "interactions": true}`).
- **Seeds.** Any endpoint that randomizes accepts an optional integer `seed`. The seed
  actually used is always echoed in the returned design's `meta` (the library already
  draws a concrete seed when none is given), so every response is reproducible.
- **Non-finite numbers.** JSON has no NaN/Infinity. NaN (saturated-model standard
  errors, Scheffé `effects`, undefined F/p) serializes as `null`; the accompanying
  `warnings` array says why.
- **Warnings are data.** Responses carry `"warnings": [ ... ]` (possibly empty) for
  non-fatal conditions: `"saturated_model"`, `"no_pure_error"` (lack-of-fit skipped),
  `"search_not_converged"`, etc. Warnings never change the HTTP status.

## Shared schema components

Pydantic models, mirroring the serialization schema:

```text
Factor            = ContinuousFactor | CategoricalFactor | MixtureFactor   # discriminated on "type"
ContinuousFactor  = {type: "continuous", name, low, high, units?}
CategoricalFactor = {type: "categorical", name, levels: [str, ...]}
MixtureFactor     = {type: "mixture", name, low?, high?}                   # proportions in [0, 1]

DesignDocument    = {schema_version, name?, factors: [Factor, ...],
                     runs: [{column: value, ...}, ...], point_types?, meta?}

ModelSpec         = "linear" | "quadratic" | "scheffe-linear" | "scheffe-quadratic"
                  | {order: 1 | 2, interactions: bool}

Bounds            = [low, high]                        # coded, applied to every factor
                  | {factor_name: [low, high], ...}    # natural units; omitted factors
                                                       # default to their full tested range
```

## Design generation

One endpoint per generator, `{factors: [Factor, ...], ...params}` in,
`{design: DesignDocument, warnings: []}` out. Parameters are the library keywords:

| Endpoint | Wraps | Parameters |
| --- | --- | --- |
| `POST /v1/designs/full-factorial` | `full_factorial` | `levels: int \| [int, ...] = 2` |
| `POST /v1/designs/fractional-factorial` | `fractional_factorial` | `generators: ["D=ABC", ...]` |
| `POST /v1/designs/plackett-burman` | `plackett_burman` | — |
| `POST /v1/designs/definitive-screening` | `definitive_screening` | `extra_center_runs = 0`, `fake_factors?` |
| `POST /v1/designs/central-composite` | `central_composite` | `alpha: "faced" \| "rotatable" \| "orthogonal" \| float = "faced"`, `center = 4`, `fraction?: [str, ...]` |
| `POST /v1/designs/box-behnken` | `box_behnken` | `center = 3` |
| `POST /v1/designs/space-filling` | `latin_hypercube` / `sobol` / `halton` | `sampler: "lhs" \| "sobol" \| "halton"`, `n_runs`, `criterion?` (lhs), `scramble = true` (sobol/halton), `seed?` |
| `POST /v1/designs/simplex-lattice` | `simplex_lattice` | `degree: int` |
| `POST /v1/designs/simplex-centroid` | `simplex_centroid` | — |
| `POST /v1/designs/extreme-vertices` | `extreme_vertices` | `include_centroid = true` |

Worked example:

```text
POST /v1/designs/central-composite
{
  "factors": [
    {"type": "continuous", "name": "temp", "low": 20, "high": 80, "units": "C"},
    {"type": "continuous", "name": "time", "low": 0, "high": 10}
  ],
  "alpha": "rotatable",
  "center": 5
}
→ 200
{
  "design": { "schema_version": "1.0", "factors": [...], "runs": [...],
              "point_types": [...], "meta": {"generator": {...}, "alpha": 1.414, ...} },
  "warnings": []
}
```

### Optimal designs

The coordinate-exchange engine gets its own shape because it returns a search report
(`OptimalDesign`) alongside the design, and because constrained regions arrive as
explicit candidate sets.

```text
POST /v1/designs/optimal
{
  "factors": [
    {"type": "continuous", "name": "temp", "low": 20, "high": 80},
    {"type": "continuous", "name": "time", "low": 0, "high": 10}
  ],
  "n_runs": 15,
  "model": "quadratic",              # default
  "criterion": "D",                  # "D" | "I", default "D"
  "n_restarts": 20,                  # default; capped, see Limits
  "max_iter": 100,
  "seed": 42,
  "region": [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],   # optional (m, k) coded candidate
             [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],       # rows; defaults to
             [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]       # candidate_grid / mixture_candidates
}
→ 200
{
  "design": DesignDocument,          # search meta (criterion, score, seed, ...) inside
  "search": {"criterion": "D", "score": 11.697910139183666, "d_efficiency": 0.46841598958482156,
             "n_restarts": 20, "converged": true},
  "warnings": []
}
```

```text
POST /v1/designs/augment
{ "design": DesignDocument, "n_runs": 8, "model": "quadratic",
  "criterion": "D", "seed": 7, "region": [...]? }
→ 200   same shape; existing rows tagged point_type "existing", new rows "augment"
```

The constrained-region workflow (WORKFLOW6) needs the candidate sets themselves, so the
client can filter them against its own feasibility rule before passing `region` back:

```text
POST /v1/designs/candidates
{ "factors": [...], "levels": 3 }                     # or resolution for mixtures
→ 200 { "points": [[...], ...], "kind": "grid" | "mixture" }
```

Candidate points are coded (`candidate_grid` convention); mixture candidates are
proportions, which are their own coded form.

### Split-plot and blocking designs (Phase 5)

Same `{factors, ...params}` in, `{design, warnings}` out as the generators above.
Split-plot designs mark their whole-plot (hard-to-change) factors with
`"hard_to_change": true` on the factor and return a `whole_plots` array on the design
document (one integer plot id per run — see [SERIALIZATION.md](SERIALIZATION.md)); the
`fit-gls` analysis endpoint consumes that structure.

| Endpoint | Wraps | Parameters |
| --- | --- | --- |
| `POST /v1/designs/split-plot` | `split_plot` | `whole_plot_design: "full" \| DesignDocument = "full"`, `sub_plot_design: "full" \| DesignDocument = "full"`, `n_whole_plot_reps = 1`, `seed?`. Whole-plot stratum = the factors flagged `hard_to_change`; the rest are the sub-plot stratum. |
| `POST /v1/designs/randomized-complete-block` | `randomized_complete_block` | one of `factors: [Factor, ...]` **or** `n_treatments: int`, plus `n_blocks`, `seed?` |
| `POST /v1/designs/latin-square` | `latin_square` | `treatments: int`, `seed?` |
| `POST /v1/designs/blocked-factorial` | `blocked_factorial` | `block_generators: ["ABC", ...]`, `seed?` — the confounded set (generators + generalized interactions) is recorded in the design's `meta["confounded_with_blocks"]` |

The block-carrying generators add a reserved `block` categorical factor to the returned
design; the existing deviation coding fits it with no analysis-side change.

Two-level **categorical** factors need no new endpoint — `POST /v1/designs/definitive-screening`
accepts them directly (it forwards the factor list), building the Jones–Nachtsheim (2013)
DSD-augment when categoricals are present. Only its run count changes; the response shape
is identical.

Worked example:

```text
POST /v1/designs/split-plot
{
  "factors": [
    {"type": "continuous", "name": "temp", "low": 100, "high": 200, "hard_to_change": true},
    {"type": "continuous", "name": "conc", "low": 1, "high": 5}
  ],
  "seed": 1
}
→ 200
{
  "design": { "schema_version": "1.0", "factors": [...], "runs": [...],
              "whole_plots": [0, 0, 1, 1], "meta": {...} },
  "warnings": []
}
```

## Design operations

All take and return a full design document — pure transformations:

| Endpoint | Wraps | Parameters |
| --- | --- | --- |
| `POST /v1/designs/validate` | `validate_design_dict` | `check_ranges = false`. Returns `{valid: bool, errors: [str, ...]}` with **200** either way — validation *outcome* is this endpoint's payload, unlike everywhere else, where an invalid design is a 422. |
| `POST /v1/designs/randomize` | `Design.randomize` | `seed?` |
| `POST /v1/designs/replicate` | `Design.replicate` | `n`, `each = false` |
| `POST /v1/designs/project` | `Design.project` | `factors: [name, ...]` — the post-screening "keep the survivors" step |
| `POST /v1/designs/responses` | `Design.with_responses` | `responses: {name: [values, ...], ...}` — values aligned to run order; length-checked |

## Analysis

Every analysis endpoint takes `{design, response, model}` and re-fits internally — OLS
is milliseconds, and re-fitting from inputs is the serialization doctrine. No fit
handle ever crosses the wire.

The worked examples below are real: a quadratic fit of a 2-factor central-composite
design (`temp` 20–80 °C, `time` 0–10 min, a synthetic `yield` response), reproduced
through the actual service and locked in as golden request/response pairs under
`doe-service/tests/contract/pairs/` (Milestone 6, `docs/WEBSERVICE_BUILD.md` §6).

```text
POST /v1/analysis/fit
{ "design": DesignDocument, "response": "yield", "model": "quadratic",
  "confidence": 0.95 }               # optional, for the CI columns
→ 200
{
  "terms": [
    { "term": "temp", "coefficient": 2.133, "effect": 4.267,
      "std_error": 0.093, "t": 22.95, "p": 7.55e-08,
      "ci_low": 1.914, "ci_high": 2.353 },
    ...
  ],
  "r_squared": 0.9966, "adjusted_r2": 0.9942, "dof_resid": 7, "mse": 0.0518,
  "fitted": [...], "residuals": [...],
  "model": {"order": 2, "interactions": true},   # the resolved spec, echoed
  "warnings": []
}
```

Scheffé fits return `effect: null` per term (the ±1 swing is meaningless on
proportions); saturated fits return `null` inference columns plus
`"saturated_model"` in `warnings`.

```text
POST /v1/analysis/fit-gls
{ "design": DesignDocument, "response": "y", "model": "linear",
  "confidence": 0.95 }               # split-plot design: must carry "whole_plots"
→ 200
{
  "terms": [ { "term": "Intercept", "coefficient": 50.625, ... },
             { "term": "temp", "coefficient": -0.625, ... }, ... ],
  "r_squared": ..., "adjusted_r2": ..., "dof_resid": ..., "mse": ...,
  "fitted": [...], "residuals": [...],
  "model": {"order": 1, "interactions": true},
  "sigma2_wp": 0.25,                 # whole-plot variance component (REML)
  "n_whole_plots": 4,
  "dof_terms": {"Intercept": 2, "temp": 2, "conc": 2, "temp:conc": 2},  # containment df
  "warnings": []
}
```

`fit-gls` is the split-plot front door: it re-fits by REML/GLS and returns everything
`fit` does plus the whole-plot variance component and the two-stratum degrees of freedom
(whole-plot terms get the coarser whole-plot df — the standard errors OLS understates).
A design with no `whole_plots` structure is a 422 `infeasible`.

```text
POST /v1/analysis/anova
{ "design": ..., "response": ..., "model": ... }
→ 200
{
  "rows": [ {"term": "temp", "ss": 27.31, "df": 1, "ms": 27.31, "f": 526.9, "p": 7.55e-08},
            ..., {"term": "Residual", ...}, {"term": "Total", ...} ],
  "lack_of_fit": {"ss_lof": 0.276, "df_lof": 3, "ss_pe": 0.087, "df_pe": 4,
                  "f": 4.23, "p": 0.099},        # null + "no_pure_error" warning
                                                  # when the design has no replicates
  "press": 2.13, "predicted_r2": 0.980,
  "warnings": []
}
```

```text
POST /v1/analysis/predict
{ "design": ..., "response": ..., "model": ...,
  "points": [ {"temp": 55, "time": 4.2}, {"temp": 20, "time": 10} ] }   # natural units
→ 200 { "predictions": [70.07, 63.04] }
```

`predict` is the workhorse for grid searches over constrained regions (WORKFLOW5/6);
`points` records must cover every factor.

```text
POST /v1/analysis/diagnostics
{ "design": ..., "model": {"order": 1, "interactions": true}, "region": [...]? }
→ 200
{
  "efficiency": {"d": 0.506, "a": 0.466, "g": 0.466, "i": 0.521},
  "condition_number": 1.80,
  "vif": {"temp": 1.0, "time": 1.0, "temp:time": 1.0},
  "correlation_matrix": {"labels": [...], "matrix": [[...], ...]},
  "leverage": [...]
}
```

No `response` — these judge the design against a model before any experiment runs.

```text
POST /v1/analysis/coverage
{ "design": ..., "method": "CD" }     # "CD" | "WD" | "MD" | "L2-star"
→ 200 { "discrepancy": 0.0457, "maximin_distance": 0.0 }
```

## Optimization

The `/optimize/*` examples below reuse the same `/analysis/fit` golden design (its
quadratic `yield` surface peaks inside the tested box); the `/desirability` example
adds two more responses (`impurity_pct`, `cost`) to the same factors. All are locked in
`doe-service/tests/contract/pairs/`.

```text
POST /v1/optimize/stationary-point
{ "design": ..., "response": "yield", "model": "quadratic" }
→ 200
{ "kind": "maximum",                  # "maximum" | "minimum" | "saddle"
  "natural": {"temp": 62.44, "time": 7.25}, "coded": [0.415, 0.449],
  "response": 70.86, "eigenvalues": [-3.40, -1.84], "warnings": [] }
```

```text
POST /v1/optimize/optimum
{ "design": ..., "response": "yield", "model": "quadratic",
  "maximize": true }                  # Bounds, see conventions; omitted here
→ 200                                 # defaults to the full coded box [-1, 1]
{ "natural": {"temp": 62.44, "time": 7.25}, "coded": [0.415, 0.449], "response": 70.86,
  "maximize": true, "at_bound": false, "warnings": [] }
```

```text
POST /v1/optimize/desirability
{
  "design": DesignDocument,           # one design carrying all response columns
  "goals": [
    { "response": "yield_pct",    "model": "quadratic", "goal": "max",
      "low": 60, "high": 90, "weight": 1.0 },
    { "response": "impurity_pct", "model": "quadratic", "goal": "min",
      "low": 0.5, "high": 3.0 },
    { "response": "cost",         "model": "linear", "goal": "target",
      "low": 8, "target": 10, "high": 12 }
  ]
}
→ 200
{ "natural": {"temp": 49.17, "time": 4.59}, "coded": [-0.028, -0.083], "overall": 0.512,
  "responses": {"yield_pct": 69.35, "impurity_pct": 1.92, "cost": 10.0},
  "individual": {"yield_pct": 0.312, "impurity_pct": 0.431, "cost": 1.0},
  "warnings": [] }
```

Each goal is `ResponseGoal.to_dict()` plus the `response`/`model` needed to re-fit its
`FitResult` server-side (the goal serialization deliberately omits the fit). All goals
share the design's factors, which the library already requires.

## Plot data

Headless cores as JSON for frontend rendering (Plotly/Vega); no matplotlib in the
service. Pareto / main-effects / half-normal plots need no endpoint — they are direct
renderings of `/v1/analysis/fit`'s `terms` array.

| Endpoint | Wraps | Request (beyond `design`/`response`/`model`) | Response |
| --- | --- | --- | --- |
| `POST /v1/plot-data/surface` | `surface_grid` | `x`, `y`, `fixed?: {factor: value}`, `resolution = 25` | `{x: [[...]], y: [[...]], z: [[...]]}` natural-unit meshes |
| `POST /v1/plot-data/interactions` | `interaction_lines` | `x`, `trace`, `fixed?`, `trace_levels?`, `resolution = 25` | `{x: [...], lines: [{trace_value, z: [...]}, ...]}` |
| `POST /v1/plot-data/ternary` | `ternary_grid` | `resolution = 100` | `{x: [...], y: [...], z: [...], points: [[a, b, c], ...]}` |
| `POST /v1/plot-data/alias` | `alias_matrix` | no `response`; `model`, `absolute = false` | `{labels: [...], matrix: [[...], ...]}` |

## Errors

One envelope for every non-2xx response. The example below is real (a 10-run document
with run 3 missing `time` and a 9-entry `point_types`), locked in
`doe-service/tests/contract/pairs/validation_error.json`:

```json
{
  "error": {
    "code": "validation_error",
    "message": "design document is invalid",
    "errors": ["run[3] missing value for factor 'time'",
               "'point_types' has 9 entries but there are 10 runs"]
  }
}
```

| Status | `code` | Trigger |
| --- | --- | --- |
| 422 | `validation_error` | Pydantic shape failure, or `ValidationError` from `validate_design_dict` — `.errors` (already exhaustive, collected in one pass) becomes the `errors` array |
| 422 | `infeasible` | Library `ValueError` with a domain message, passed through verbatim: Sobol non-power-of-two run counts, infeasible mixture bounds, mixed mixture/non-mixture factor sets, `x == y` in surface grids, bad goal brackets, ... |
| 422 | `limit_exceeded` | A parameter cap (below) was hit; message names the cap and the ceiling |
| 400 | `malformed` | Body is not valid JSON |
| 500 | `internal` | Anything else; no library detail leaks |

The library's error messages are written to be user-facing (they name nearest valid
Sobol sizes, list all validation problems, etc.) — the service's job is to pass them
through, not rewrite them.

## Limits

Caps keep every endpoint synchronous in v1 (WEBSERVICE.md's option (a)); the async job
pattern is deferred until these chafe. All configurable at deployment; proposed
defaults:

| Cap | Default | Guards |
| --- | --- | --- |
| factors per design | 32 | model-matrix width |
| runs per design | 10 000 | payload size, OLS cost |
| `n_restarts` × `max_iter` (optimal) | 20 × 100 | coordinate-exchange wall time |
| `region` / candidate rows | 100 000 | exchange sweep cost |
| grid `resolution` (surface/ternary) | 200 | response payload |
| desirability goals | 8 | `differential_evolution` dimensionality is fixed, but each goal is a re-fit |
| request body | 5 MB | everything |

## Implementation notes

- The service lives in the `doe-service` uv workspace member of this repository
  (decided; see [WEBSERVICE.md](WEBSERVICE.md) "Packaging"). FastAPI, one router per
  section above:

  ```text
  doe-service/src/doe_service/
      main.py            # app factory, error handlers, limits config
      schemas/           # pydantic: factors, design document, model spec, results
      routers/
          designs.py     # generation + operations + candidates
          analysis.py
          optimize.py
          plot_data.py
      limits.py
  ```

- `Factor` is a Pydantic discriminated union on `"type"`, mirroring `factor_from_dict`.
  Inbound design documents go Pydantic → `validate_design_dict` → `Design.from_dict`;
  outbound designs are `Design.to_dict()` verbatim (already JSON-safe — numpy scalars
  are coerced on the way out).
- The analysis/optimization response shapes above are exactly the `to_dict`
  serializations that WEBSERVICE.md phase 1 adds to the library (`FitResult`,
  ANOVA rows, `LackOfFit`, `StationaryPoint`, `Optimum`, `DesirabilityResult`,
  `Efficiency`). The service must not assemble these by hand from dataclass fields —
  the shapes live in the library, tested there, and the routers stay one-call thin.
- Contract tests mirror the library's style: golden request/response JSON pairs for
  known designs (the 2^(4-1) defining relation, textbook CCD run counts) so the wire
  format is anchored the same way the library is.

## Open questions

- **Richer response attachment.** `POST /v1/designs/responses` takes plain aligned
  arrays for now; the units/status/QC-flag readout document (SERIALIZATION.md roadmap
  item 2) and stable `run_id` joins should replace positional alignment once roadmap
  item 1 lands.
- **Auth.** Out of scope for the spec; WEBSERVICE.md phase 3 proposes API keys. Nothing
  above depends on identity.
- **Async jobs.** If the optimal-design caps prove too tight, `POST /v1/designs/optimal`
  gains a `202 {job_id}` mode and a `GET /v1/jobs/{id}`; the request schema is designed
  so this is additive.
- **Region points on the wire.** Coded, matching `candidate_grid`. Revisit if
  natural-unit constraint specs (e.g. `"temp + 10*conc <= 90"`) turn out to be the
  common client need — a `constraints` DSL evaluated server-side over the grid would
  replace client-side filtering.
