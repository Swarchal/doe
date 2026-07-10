# Web service plan

A plan for exposing the DoE library as a backend to a web service — interacting with it
via an API over HTTP. This weighs the architectural options and records the recommended
path; nothing here is implemented yet.

The library was built with this in mind, and the head start is real: a versioned JSON
design document (`Design.to_dict()`, [SERIALIZATION.md](SERIALIZATION.md)) that is the
natural wire format, `validate_design_dict` as a ready-made request validator, the
"analysis results are derived — store inputs and re-fit" doctrine, concrete recorded
seeds making generation reproducible (and cacheable), and headless plot cores
(`surface_grid`, `interaction_lines`, `ternary_grid`, `alias_matrix`) that return data a
frontend can render without matplotlib in the loop.

## The central decision: stateless vs stateful

This matters more than which framework or protocol is chosen.

### Option 1 — stateless compute service

Every request carries its full inputs (a design dict, generator parameters, a response
column name) and every response carries full outputs. No database, no sessions.

```text
POST /v1/designs/central-composite   {factors, alpha, center}      → design dict
POST /v1/analysis/fit                {design, response, model}     → fit summary JSON
```

**Pros**

- Trivially scalable and cacheable; nothing to migrate or back up.
- The wire format *is* the existing `Design.to_dict()` schema — the serialization work
  is largely done, and `validate_design_dict` already collects all problems in one pass
  (perfect for a single 422 response body).
- Testing is pure input → output, matching the library's existing test style.
- Clients (a web UI, automation software, a notebook) hold state however they like.
- The SERIALIZATION.md doctrine makes this natural: OLS is milliseconds, so an endpoint
  that needs a fitted model takes `design + response` and re-fits internally rather
  than referencing a persisted `FitResult`.

**Cons**

- Payloads travel repeatedly (in practice fine — designs are tens of rows, kilobytes).
- No experiment lifecycle: no "list my experiments", no attaching readouts over days,
  no audit trail.
- Multi-step flows (screen → `project` → `augment`) mean the client shuttles the design
  dict between calls.

### Option 2 — stateful experiment service

Designs/experiments are persisted resources with IDs:

```text
POST /v1/experiments                     → create, returns id
POST /v1/experiments/{id}/responses      → attach readouts as they arrive
GET  /v1/experiments/{id}/analysis       → fit on demand
```

**Pros**

- Matches the real workflow: an experiment lives for days between design and readout.
- Enables collaboration, history, provenance — the `run_id`-joined readout ingestion
  already sketched in the SERIALIZATION.md roadmap.
- The natural backend for a web *app* rather than a web *API*.

**Cons**

- The service now owns a database, migrations, auth/tenancy, concurrency (two people
  attaching responses), and lifecycle semantics — most of the complexity ends up in
  storage plumbing rather than in DoE.

### Recommendation: build 1, design for 2

A stateless compute core is useful immediately and never wasted — a later stateful
layer is a thin CRUD shell that stores design documents and *calls the same compute
endpoints/functions*. Going stateful first couples the statistics to storage decisions
there are no requirements for yet.

## API style

| Style | Verdict |
| --- | --- |
| REST-ish JSON over HTTP | **Recommended.** Fits the domain: documents in, documents out. |
| GraphQL | Poor fit. Wins when clients slice varied views of a big object graph; this is compute operations on self-contained documents. Adds a schema layer for nothing. |
| gRPC | Wrong audience. Great for internal service-to-service, hostile to browsers and curl-from-a-lab-PC. Revisit only if a robot-orchestration platform demands it. |
| JSON-RPC / plain RPC | Honest about the semantics but loses the HTTP ecosystem (status codes, OpenAPI tooling, caching). |
| MCP server | Not a replacement, but a cheap *complement*: the same compute functions exposed as MCP tools would let AI agents drive DoE workflows. Same core, second thin adapter. |

Within REST, resource purism isn't worth contorting for. Design generation is naturally
RPC-shaped — embrace `POST /v1/designs/{generator}` and `POST /v1/analysis/fit` rather
than inventing fake resources. One idempotency bonus: every generator records a concrete
seed in `meta`, so identical requests with a given seed return identical designs —
genuinely cacheable.

## Framework

**FastAPI** is the clear pick:

- Pydantic request/response models can mirror the serialization schema — a
  machine-checked, OpenAPI-published version of the design-dict contract for free.
  Pydantic checks *shape*; `validate_design_dict` keeps running underneath as the
  *semantic* layer (cross-field consistency such as `point_types` alignment and
  categorical values drawn from declared levels).
- Auto-generated OpenAPI/Swagger docs matter for a scientific API whose users are lab
  scientists and integrators, not web developers.
- Async support is available when the job-polling pattern (below) becomes necessary.

Alternatives considered: Flask is fine but validation and docs would be hand-rolled;
Django REST buys an ORM/admin not needed until the stateful layer, and even then a
lighter store may do; Litestar is credible but FastAPI's ecosystem wins for an open
scientific tool.

## Packaging: a workspace member, not an extra

**Decided (July 2026): the service is a separate distribution, `doe-service`, living in
this repository as a uv workspace member** (`doe-service/`, depending on the in-tree
`doe` via `[tool.uv.sources]`). Not inside `doe`, and not an optional extra:

- An HTTP layer breaks the scipy-stack dependency constraint in *kind*, not just count —
  FastAPI/Pydantic/uvicorn are a different ecosystem with a different release tempo, and
  a `doe` release should never be blocked on a web-framework migration (or vice versa).
- Most `doe` users are notebook/script users who will never run a server; application
  code inside the library muddies the public API. The `plotting` extra works as an
  in-package extra because it is a thin module behind a lazy import — a service
  (config, error handlers, deployment) is an application, not a module.
- The JSON contracts stay in `doe` (see below); pushing them into the service would make
  the service the de-facto owner of the wire format.

Same repo rather than a separate one, for now: schema evolution is the dominant
coupling early on (`run_id`s and the analysis `to_dict`s land in `doe` while the service
is built against them), so atomic cross-package commits and single-CI contract testing
beat coordinating two repos' releases. The split-out option stays open — and stays a
cheap `git mv` — under one discipline: **the dependency points one way only.**
`doe_service` imports `doe`; `doe` never imports `doe_service`.

Development commands, from `doe-service/`:

```bash
uv run --extra dev pytest        # service tests
uv run --extra dev mypy          # strict, like the root package
uv run --extra dev uvicorn --factory doe_service.main:create_app --reload
```

## Endpoint sketch (v1, stateless)

This is the birds-eye view; the concrete request/response contracts are specified in
[WEBSERVICE_API.md](WEBSERVICE_API.md).

```text
# generation — one endpoint per generator family
POST /v1/designs/full-factorial
POST /v1/designs/fractional-factorial
POST /v1/designs/plackett-burman
POST /v1/designs/definitive-screening
POST /v1/designs/central-composite
POST /v1/designs/box-behnken
POST /v1/designs/optimal              # d_optimal / i_optimal / augment via criterion + fixed rows
POST /v1/designs/space-filling        # latin_hypercube / sobol / halton via sampler
POST /v1/designs/mixture              # simplex_lattice / simplex_centroid / extreme_vertices

# design operations
POST /v1/designs/validate             # expose validate_design_dict for pre-flight checks
POST /v1/designs/randomize
POST /v1/designs/replicate
POST /v1/designs/project

# analysis — all take {design, response, model spec}, re-fit internally
POST /v1/analysis/fit                 # FitResult summary
POST /v1/analysis/anova               # anova_table + lack_of_fit
POST /v1/analysis/diagnostics         # efficiency, vif, condition_number, correlation_matrix, leverage
POST /v1/analysis/coverage            # discrepancy, maximin_distance

# optimization
POST /v1/optimize/stationary-point
POST /v1/optimize/optimum
POST /v1/optimize/desirability        # goals serialized via ResponseGoal.to_dict

# plot data — headless cores as JSON for frontend rendering
POST /v1/plot-data/effects            # pareto / half-normal / main-effects inputs
POST /v1/plot-data/interactions       # interaction_lines
POST /v1/plot-data/surface            # surface_grid (contour + 3-D surface)
POST /v1/plot-data/ternary            # ternary_grid
POST /v1/plot-data/alias              # alias_matrix
```

## Design details that will bite

**Long-running operations.** `fit_ols`, `anova_table`, and the named generators run in
milliseconds — plain synchronous endpoints. But `coordinate_exchange` with many restarts
over a big candidate grid, and `desirability`'s `differential_evolution`, can run
seconds to minutes. Two options: (a) cap parameters (`n_restarts`, grid resolution) and
stay synchronous with a request timeout; (b) the async job pattern — return `202` plus a
job ID, client polls `GET /v1/jobs/{id}`. Start with (a) plus honest documented limits;
add (b) only when someone hits the ceiling. No Celery/Redis on day one — FastAPI
background tasks and an in-process job table cover a single-node service.

**Plots.** Two paths, and the library already supports the better one:

- *Server-rendered images* (`.../pareto.png` via matplotlib Agg): simple for report
  clients, but poor for interactive UIs, and matplotlib-in-a-server brings thread-safety
  and font-cache headaches.
- *Headless data endpoints*: return the plot cores as JSON and let the frontend render
  with Plotly/Vega — interactive, cacheable, no matplotlib in the service.

Ship the data endpoints as primary; add PNG rendering later as a convenience if
report-generation clients want it.

**Analysis output schemas — the real library gap.** `Design` round-trips beautifully,
but `FitResult`, `anova_table` (a DataFrame), `LackOfFit`, `StationaryPoint`, `Optimum`,
`DesirabilityResult`, and `Efficiency` have no `to_dict`. This is the main *library*
work the service needs: a JSON shape for each analysis result. One-way is enough — these
are derived outputs, so no `from_dict` is required (beyond what `ResponseGoal` already
has). Doing it in the library rather than the service layer keeps the contract testable
with the existing test pattern and reusable by a future MCP adapter.

**Errors and warnings.** Map `ValidationError.errors` to a 422 whose body lists every
problem (it already collects them all). The saturated-model case *warns* and returns
NaN standard errors — surface that as a `warnings` field in the fit response, not an
error. Infeasibility errors (Sobol power-of-two run counts, mixture bound feasibility)
raise with helpful messages that should pass through to the client verbatim.

**Versioning.** `schema_version` already versions the *document*; version the API path
(`/v1/...`) independently, since endpoint shapes and the document schema will evolve at
different rates.

## Phased plan

1. **Library prep (no HTTP yet).** `to_dict` for the analysis result types listed
   above; land SERIALIZATION.md roadmap item 1 (stable `run_id`s), since both the
   stateless response-attachment flow and any future stateful layer join on it.
2. **Stateless FastAPI service** in the `doe-service` workspace package: the endpoint sketch above,
   Pydantic models mirroring the design-dict schema, OpenAPI docs, parameter caps on
   the optimizers.
3. **Hardening.** Auth (API keys are enough initially), rate limits on the expensive
   endpoints, the async job pattern if the parameter caps chafe.
4. **Stateful layer (when a real UI or lab need exists).** Experiments as resources
   over a small database, storing the same JSON documents, readout attachment per
   SERIALIZATION.md roadmap item 2, calling the phase-2 compute unchanged. Optionally
   an MCP adapter over the same core.

The one-sentence version: a stateless FastAPI service whose wire format is the
`Design.to_dict()` schema the library already ships, with the main prerequisite being
`to_dict` coverage for the analysis result types — and persistence deliberately
deferred until there is a concrete consumer for it.
