# doe-service — implementation build plan

**Status: all six milestones are done.** Every v1 route is implemented (no 501 stubs
remain), every WEBSERVICE_API.md example is a passing contract test under
`doe-service/tests/contract/`, parameter caps are enforced across every endpoint plus a
request body-size limit, and the OpenAPI schema documents all 31 operations (30 POST +
`GET /v1/health`) with typed request/response models. This doc is kept as the historical
build record; see WEBSERVICE.md for the current status summary.

Detailed build plan for implementing the v1 API specified in
[WEBSERVICE_API.md](WEBSERVICE_API.md), inside the `doe-service` workspace package
(packaging decision in [WEBSERVICE.md](WEBSERVICE.md)). The skeleton — app factory,
`/v1/health`, stub routers returning 501, module layout — already exists in
`doe-service/`; this plan turns each stub into the specified contract, milestone by
milestone.

Two rules carry through every milestone:

- **Shapes live in the library.** Every response body is a `to_dict` (or records
  helper) implemented and tested in `doe`, not assembled from dataclass fields in a
  router. Routers stay one-call thin: parse → call → serialize.
- **The dependency points one way.** `doe_service` imports `doe`; `doe` never imports
  `doe_service`. Milestone 0 is `doe` work and ships independently.

## Goals (definition of done for v1)

1. Every route in the WEBSERVICE_API.md tables is implemented, and the stub-era
   `tests/test_routes.py` route-table test still passes unchanged.
2. Every request/response example in WEBSERVICE_API.md is a passing **contract test**
   (golden JSON in, golden JSON out, anchored the same way the library anchors known
   designs).
3. `uv run --extra dev mypy` (strict) and `pytest` pass in `doe-service/`; root
   `ruff check .` passes; no endpoint returns a body containing `NaN`.
4. The OpenAPI schema at `/openapi.json` documents every endpoint with typed request
   models — no `dict[str, Any]` bodies left.

## What's new / what changes

```
src/doe/                              # Milestone 0 (library prep)
  serialization.py    # NEW: json_safe() — numpy→native + NaN/Inf→None, shared by all to_dicts
  analysis/
    fit.py            # NEW: FitResult.to_dict(confidence=0.95); SaturatedFitWarning category
    anova.py          # NEW: anova_records(result); LackOfFit.to_dict()
    diagnostics.py    # NEW: Efficiency.to_dict()
    optimize.py       # NEW: StationaryPoint/Optimum/DesirabilityResult .to_dict()

doe-service/src/doe_service/          # Milestones 1–6
  main.py             # CHANGED: error handlers + limits wired in (routers already mounted)
  errors.py           # M1: envelope model, ServiceError, register_exception_handlers
  limits.py           # exists (defaults from the spec); M6 enforces everywhere
  convert.py          # M1: design_from_document, resolve_model, captured_warnings, jsonable
  schemas/
    factors.py        # M1: Factor discriminated union (mirrors factor_from_dict)
    design.py         # M1: DesignDocument, DesignResponse
    common.py         # M1: ModelSpec, Bounds, request bases
    results.py        # M3+: typed response models (from the library to_dict shapes)
  routers/
    designs.py        # M2: generation + operations + candidates (18 routes)
    analysis.py       # M3: fit, anova, predict, diagnostics, coverage
    optimize.py       # M4: stationary-point, optimum, desirability
    plot_data.py      # M5: surface, interactions, ternary, alias
doe-service/tests/
  test_routes.py      # exists: the full v1 route table is mounted
  test_designs.py     # M2   } unit tests per router, plus
  test_analysis.py    # M3   } tests/contract/ golden request/response
  test_optimize.py    # M4   } JSON pairs (M6)
  test_plot_data.py   # M5
```

## 0. Milestone 0 — library prep (in `doe`) — done

Ships as an ordinary `doe` change with its own tests; nothing imports the service.

### 0.1 `json_safe` — `serialization.py`

```python
def json_safe(value: Any) -> Any:
    """Recursively coerce to JSON-serializable: numpy scalars/arrays -> native,
    NaN/Inf -> None, mappings/sequences walked."""
```

The single place the "JSON has no NaN" rule lives. `Design.to_dict` already coerces
numpy scalars; it switches to this helper. Every new `to_dict` below returns
`json_safe`d output, so the service never has to sanitize.

### 0.2 `FitResult.to_dict` — `analysis/fit.py`

```python
def to_dict(self, *, confidence: float = 0.95) -> dict[str, Any]:
```

Exactly the `/v1/analysis/fit` response body minus `warnings`: `terms` (one record per
term — `term`, `coefficient`, `effect`, `std_error`, `t`, `p`, `ci_low`, `ci_high`,
from `summary()` + `conf_int(confidence)`), `r_squared`, `adjusted_r2`, `dof_resid`,
`mse`, `fitted`, `residuals`, and the resolved `model: {order, interactions}`.
Saturated fits serialize their NaN inference columns as `null`; Scheffé fits serialize
`effect: null` (also `adjusted_r2`/`mse` when `dof_resid == 0`).

Also: `class SaturatedFitWarning(UserWarning)` and `fit_ols` warns with that category
instead of bare `UserWarning` — backwards compatible, and it lets the service map the
warning to the string `"saturated_model"` by category rather than by message matching.

### 0.3 ANOVA serialization — `analysis/anova.py`

```python
def anova_records(result: FitResult) -> list[dict[str, Any]]:   # module-level
class LackOfFit:
    def to_dict(self) -> dict[str, Any]: ...
```

`anova_records` maps the `anova_table` DataFrame to the spec's row records — columns
rename `SS/df/MS/F/p` → `ss/df/ms/f/p`, index becomes `term`, `Residual`/`Total` rows
included, NaN F/p (on the Residual/Total rows) → `null`. `LackOfFit.to_dict` is the
flat field dump (`ss_lof`, `df_lof`, `ss_pe`, `df_pe`, `f_stat` → `f`, `p_value` → `p`).

### 0.4 Optimization / diagnostics results — `analysis/optimize.py`, `analysis/diagnostics.py`

`StationaryPoint.to_dict` (`kind`, `natural`, `coded`, `response`, `eigenvalues`,
`eigenvectors`, `response_name`), `Optimum.to_dict` (`natural`, `coded`, `response`,
`maximize`, `at_bound`, `response_name`), `DesirabilityResult.to_dict` (`natural`,
`coded`, `overall`, `responses`/`individual` Series → plain dicts), and
`Efficiency.to_dict` (`d`, `a`, `g`, `i`). All via `json_safe`.

### 0.5 Tests

One test per `to_dict` reusing the existing golden anchors (e.g. the known-effects fit:
its `to_dict()["terms"]` carries the exact injected coefficients; a saturated fit
serializes `std_error: None`; `json.dumps` succeeds on every result — the NaN test).

**Deferred, not blocking:** stable `run_id`s (SERIALIZATION.md roadmap item 1). v1
attaches responses positionally via `with_responses`; `run_id` joins upgrade
`/v1/designs/responses` later without breaking it.

## 1. Milestone 1 — service foundations — done

### 1.1 Schemas — `schemas/factors.py`, `schemas/design.py`, `schemas/common.py`

Pydantic v2, mirroring the serialization schema (shapes in WEBSERVICE_API.md "Shared
schema components"):

- `ContinuousFactorSchema` / `CategoricalFactorSchema` / `MixtureFactorSchema`,
  discriminated union `FactorSchema` on `type` — field-for-field with
  `factor_from_dict`.
- `DesignDocument` — `schema_version`, `name`, `factors`, `runs` (list of
  `dict[str, Any]`; per-run key checking stays with `validate_design_dict`, which
  already does it exhaustively — don't duplicate), `point_types`, `meta`. Pydantic
  checks *shape*, the library validator checks *consistency*.
- `common.py` — `ModelSpec` (a `Literal[...] | ModelSpecObject` union), `Bounds`
  (`tuple[float, float] | dict[str, tuple[float, float]]`), and a `DesignRequest` base
  model (`design: DesignDocument`) the routers extend.
- `DesignResponse` — `{design: DesignDocument, warnings: list[str]}`, the shared
  generator/operation response.

### 1.2 Plumbing — `convert.py`

```python
def design_from_document(document: Mapping[str, Any]) -> Design:
    # validate_design_dict -> Design.from_dict; ValidationError propagates (handler maps it)
def resolve_model(model: ModelSpecLike | None, *, default: ModelSpecLike) -> tuple[int, bool]:
    # "linear"/"quadratic"/"scheffe-*"/{order, interactions} -> (order, interactions);
    # scheffe names additionally require design.factors.is_mixture (else infeasible)
def captured_warnings() -> AbstractContextManager[list[str]]:
    # warnings.catch_warnings(record=True); categories -> spec strings
    # (SaturatedFitWarning -> "saturated_model"); unknown warnings -> str(message)
def jsonable(value: object) -> object:
    # thin wrapper over doe.serialization.json_safe — the boundary safety net for
    # values that didn't come from a library to_dict (e.g. plot meshes)
```

### 1.3 Errors — `errors.py`

The envelope model (`{"error": {"code", "message", "errors"}}`) and
`register_exception_handlers(app)` (already called by `create_app`) installing, per the
spec's table:

| Exception | → status / `code` |
| --- | --- |
| `fastapi.exceptions.RequestValidationError` | 422 `validation_error`, pydantic errors flattened into `errors` |
| `doe.ValidationError` | 422 `validation_error`, `.errors` passed through |
| `ValueError` raised by `doe` | 422 `infeasible`, message verbatim |
| `LimitExceeded` (new, in `limits.py`) | 422 `limit_exceeded` |
| JSON decode failure | 400 `malformed` |
| anything else | 500 `internal`, generic message |

The `ValueError` handler is scoped to router calls (a helper `call_library(fn, ...)` or
a try/except in each router — decide in review; the helper keeps routers uniform), so a
genuine service bug still surfaces as 500, not a fake 422.

### 1.4 Definition of done

`design_from_document` and `resolve_model` unit-tested against good/bad documents; a
posted malformed design to any M2 endpoint returns the envelope with *all* validator
errors; stub routes still 501 but now behind typed request models where already known.

## 2. Milestone 2 — designs router — done

Each generation endpoint: request model = `factors: list[FactorSchema]` + the library
keywords from the WEBSERVICE_API.md table; body = build `FactorSet` → call generator →
`DesignResponse(design=result.to_dict(), warnings=captured)`. Operations endpoints take
`DesignRequest` + parameters and return the same shape.

Special cases:

- **`/optimal` and `/augment`** call `coordinate_exchange` directly (not the
  `d_optimal`/`i_optimal` wrappers) because the response needs the `OptimalDesign`
  search report (`criterion`, `score`, `d_efficiency`, `n_restarts`, `converged`)
  alongside the design. `augment` builds `fixed_runs` from the posted design's coded
  rows exactly as `doe.augment` does — or better, `doe.augment` grows a
  report-returning core the wrapper and the service share (small library follow-up;
  decide when implementing).
- **`/candidates`** dispatches on the factor set: all-mixture → `mixture_candidates`
  (`resolution` parameter, `kind: "mixture"`), otherwise `candidate_grid` (`levels`,
  `kind: "grid"`).
- **`/validate`** returns 200 with `{valid, errors}` both ways — the one endpoint where
  an invalid design is payload, not an error response.
- **`/responses`** length-checks via `with_responses` (its `ValueError` → 422
  `infeasible` naming the mismatch).
- **`region`** arrives as `list[list[float]]`, is shape-checked against `len(factors)`
  and the row cap, then passed as an ndarray.

Tests: textbook run counts through the HTTP layer (2^k, 2^(4-1) with its defining
relation, PB sizes, CCD/Box-Behnken counts, `{3,2}` lattice's six points); seed echo
(`meta` carries a concrete seed for randomize/optimal/space-filling); Sobol non-power-
of-two → 422 `infeasible` naming nearest valid sizes; validate's 200-both-ways.

## 3. Milestone 3 — analysis router (needs M0) — done

All five endpoints share a prologue: `design_from_document` → check `response` is a
run column (else 422 `validation_error` listing available columns) → `resolve_model` →
`fit_ols` inside `captured_warnings()`.

- **`/fit`** → `result.to_dict(confidence=body.confidence)` + `warnings`.
- **`/anova`** → `anova_records(result)`; `lack_of_fit` inside a try — the "needs
  replicates" `ValueError` becomes `lack_of_fit: null` + `"no_pure_error"` warning, not
  an error; `press`/`predicted_r2` null-guarded the same way for saturated fits.
- **`/predict`** — records → column mapping → `FitResult.predict`; every factor must be
  covered by every record (422 otherwise, naming the missing factor/record index).
- **`/diagnostics`** — no response column: `build_model_matrix(design, order,
  interactions)`, then `efficiency` (with optional `region`), `vif`,
  `condition_number`, `correlation_matrix`, `leverage`. Response assembled from
  `Efficiency.to_dict()` + `json_safe`d arrays.
- **`/coverage`** — `discrepancy(design, method=...)` + `maximin_distance(design)`.

Tests: fit response equals the library fit's `to_dict` for a golden design (the service
adds nothing); saturated fit → `std_error: null` + `"saturated_model"`; Scheffé fit →
`effect: null`; anova on a replicate-free design → `lack_of_fit: null` +
`"no_pure_error"`; response-name typo → 422 listing columns.

## 4. Milestone 4 — optimize router — done

- **Bounds**: the wire forms map 1:1 onto what `optimum`/`desirability` already accept
  (a coded pair, or a natural-units `{name: (low, high)}` mapping) — passthrough after
  validating factor names; unknown names → 422 `validation_error`.
- **`/stationary-point`** and **`/optimum`**: fit (quadratic required — `resolve_model`
  rejects `order=1` here with `infeasible`, matching the library's quadratic-form
  requirement) → `stationary_point(result)` / `optimum(result, maximize, bounds)` →
  `.to_dict()`.
- **`/desirability`**: per goal — fit its `response`/`model` on the shared design, then
  `ResponseGoal(result=fit, goal=..., low=..., high=..., target=..., weight=...)`
  (bracket `ValueError`s → 422 `infeasible`); then `desirability(goals, bounds=...)` →
  `.to_dict()`. Goal count capped by `limits.max_goals`.

Tests: WORKFLOW/WORKFLOW2's known optimum and desirability points reproduced through
HTTP (the build scripts' numbers are the anchors); a `target` outside `(low, high)` →
422; bounds with an unknown factor name → 422.

## 5. Milestone 5 — plot-data router — done

Thin wrappers over the headless cores; fit prologue shared with M3.

- **`/surface`** → `surface_grid(result, x, y, fixed, resolution)` → meshes as nested
  lists (`json_safe`).
- **`/interactions`** → `interaction_lines(...)` → `{x, lines: [{trace_value, z}]}`.
- **`/ternary`** → `ternary_grid(result, resolution)` → flat `x`/`y`/`z` + `points`.
- **`/alias`** — no response column; `alias_matrix(design, order, interactions,
  absolute)` → `{labels, matrix}`.

`resolution` capped by `limits.max_resolution`. Library errors (`x == y`, categorical
axis, non-3-component ternary) pass through as 422 `infeasible`.

Tests: surface grid shape is `(resolution, resolution)` and matches a direct
`surface_grid` call; ternary on a non-mixture fit → 422.

## 6. Milestone 6 — hardening and contract lock-in — done

- **Limits enforced everywhere**: a `Limits` instance on `app.state`, checked in the
  request models (validators) or the shared prologue — factor count, run count,
  restarts×iterations, region rows, resolution, goals; plus a `Content-Length`
  middleware for `max_body_bytes` → 413 with the envelope. Each check names the cap and
  ceiling in its message.
- **Contract tests**: `tests/contract/` — one golden request/response JSON pair per
  spec example, run through `TestClient` and compared structurally. These are the wire
  format's anchor, the way the library anchors textbook designs.
- **OpenAPI polish**: router tags (already), request/response examples from the
  contract pairs, endpoint docstrings lifted from the spec.
- **Docs**: WEBSERVICE.md phase-2 status flip; service README gains a quickstart
  (generate a CCD with `curl`); CLAUDE.md status line.

## Build order

0. **M0** — library `to_dict`s + `json_safe` + `SaturatedFitWarning` (own PR, ships alone)
1. **M1** — schemas, convert, errors (service compiles against real request models)
2. **M2** — designs router (no M0 dependency — designs already serialize)
3. **M3** — analysis router (first consumer of M0)
4. **M4** — optimize router
5. **M5** — plot-data router
6. **M6** — limits enforcement, contract tests, OpenAPI/docs polish

M2 can start in parallel with M0; M3–M5 are sequential consumers of both. Each
milestone lands green: route-table test unchanged, remaining stubs still 501, mypy
strict, repo-wide ruff.

## Resolved decisions

- **Routers never touch dataclass fields** — if a response needs a shape the library
  doesn't serialize, the fix is a library `to_dict`, not router assembly.
- **`StationaryPoint.to_dict` includes `eigenvectors`** (the spec's example was
  abridged): the canonical directions are the useful half of a ridge analysis, and
  omitting them would force a second endpoint later.
- **Warning mapping is by category, not message** — hence `SaturatedFitWarning` in M0.
- **`/optimal` responds with the search report at the top level** (not buried in
  `meta`) because it answers "did the search converge, how good is it" — while `meta`
  remains the durable, serialized record.

## Deferred (unchanged from WEBSERVICE.md / WEBSERVICE_API.md)

Auth (API keys, phase 3), async jobs for optimal search (schema designed to be
additive), `run_id`-based response attachment, server-rendered PNGs, a natural-units
constraints DSL for `region`, and the stateful experiment layer.
