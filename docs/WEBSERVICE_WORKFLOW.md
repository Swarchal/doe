# End-to-end workflow over the HTTP service: from factors to an operating point

This is the [WORKFLOW.md](WORKFLOW.md) walkthrough — the same reaction-optimization study,
step for step — driven entirely through the [`doe-service`](WEBSERVICE.md) HTTP API instead
of the in-process library. If you have read WORKFLOW.md the shape is familiar; what changes
is that every step is a `POST` with a JSON body, and the thing you carry from one step to
the next is a **design document** on the wire rather than a `Design` object in memory.

The same loop as before:

1. pin down the factors and the ranges you are willing to run,
2. pick a design that can see curvature — not just slopes — and randomize the run order,
3. run the experiment and record the response against each run,
4. fit a model to the results,
5. decide whether the model can be trusted,
6. read the best setting off the fitted surface,
7. confirm it with a fresh run.

The example is the same small reaction-optimization study: three continuous factors —
temperature, reaction time, catalyst loading — and one response to maximize, percent yield.

> Every request and console output below is real, produced by
> `doe-service/scripts/build_webservice_workflow.py`, which runs the whole walkthrough
> through the actual FastAPI app. The synthetic response and the randomization are seeded,
> so the numbers are **identical to WORKFLOW.md** and reproduce exactly. This doc is
> text-only: the service is headless (it returns plot *data*, not images), so where
> WORKFLOW.md shows a figure this one names the endpoint that feeds it.

## Running it

Start the dev server from the `doe-service` directory:

```bash
uv run --extra dev uvicorn --factory doe_service.main:create_app --reload
```

The commands below use `curl` against `http://localhost:8000` and [`jq`](https://jqlang.org/)
to pass the design document from one call to the next — the wire-format equivalent of
holding onto a `Design`. Each step writes its result to a file so the next step can pick it
up. See [WEBSERVICE_EXAMPLES.md](WEBSERVICE_EXAMPLES.md) for a standalone example of every
endpoint, and [WEBSERVICE_API.md](WEBSERVICE_API.md) for the full contract.

## 1. Define the experimental space

Same knobs, same ranges as WORKFLOW.md — but here they are the JSON `factors` array that
opens almost every request. You still enter everything in natural units (°C, min, g/L); the
service rescales each factor to a common −1…+1 range internally, so the midpoint of each
range (60 °C, 40 min, 1.5 g/L) codes to zero.

```json
[
  {"type": "continuous", "name": "temperature", "low": 45, "high": 75, "units": "C"},
  {"type": "continuous", "name": "time",        "low": 20, "high": 60, "units": "min"},
  {"type": "continuous", "name": "catalyst",    "low": 0.5, "high": 2.5, "units": "g/L"}
]
```

## 2. Generate and randomize the design

To find a *best setting* rather than just rank the factors, the model has to see curvature,
so — exactly as in WORKFLOW.md — the design is a **faced central composite** with five
center replicates. Two calls: generate the design, then randomize the run order. (The
generator lays the runs down in textbook order; randomizing is its own operation on the
service, the wire equivalent of `.randomize(seed=...)`.)

```bash
# 2a. Generate the faced central-composite design.
curl -s -X POST http://localhost:8000/v1/designs/central-composite \
  -H 'Content-Type: application/json' \
  -d '{
    "factors": [
      {"type": "continuous", "name": "temperature", "low": 45, "high": 75, "units": "C"},
      {"type": "continuous", "name": "time",        "low": 20, "high": 60, "units": "min"},
      {"type": "continuous", "name": "catalyst",    "low": 0.5, "high": 2.5, "units": "g/L"}
    ],
    "alpha": "faced",
    "center": 5
  }' | jq '.design' > design_std.json

# 2b. Randomize the run order (seed echoed back in meta for reproducibility).
jq -n --slurpfile d design_std.json '{design: $d[0], seed: 20260707}' \
  | curl -s -X POST http://localhost:8000/v1/designs/randomize \
      -H 'Content-Type: application/json' -d @- \
  | jq '.design' > design.json
```

A quick look at what came back:

```bash
jq '{n_runs: (.runs | length),
     n_center: ([.point_types[] | select(. == "center")] | length),
     random_seed: .meta.random_seed,
     first_8: [.runs[:8][] | {std_order, temperature, time, catalyst}]}' design.json
```

```json
{
  "n_runs": 19,
  "n_center": 5,
  "random_seed": 20260707,
  "first_8": [
    {"std_order": 12, "temperature": 60.0, "time": 40.0, "catalyst": 0.5},
    {"std_order": 16, "temperature": 60.0, "time": 40.0, "catalyst": 1.5},
    {"std_order": 1,  "temperature": 45.0, "time": 20.0, "catalyst": 2.5},
    {"std_order": 8,  "temperature": 45.0, "time": 40.0, "catalyst": 1.5},
    {"std_order": 17, "temperature": 60.0, "time": 40.0, "catalyst": 1.5},
    {"std_order": 14, "temperature": 60.0, "time": 40.0, "catalyst": 1.5},
    {"std_order": 6,  "temperature": 75.0, "time": 60.0, "catalyst": 0.5},
    {"std_order": 13, "temperature": 60.0, "time": 40.0, "catalyst": 2.5}
  ]
}
```

The `std_order` column remembers where each run sat in the textbook layout, so any result
traces back; the row order shown is the order to run at the bench. The 19 runs are the same
three kinds as before — eight factorial corners, six axial (face) points that let the model
see curvature, and the center replicated five times, which gives a direct read on
measurement noise later. The randomization seed is echoed under `meta.random_seed`, so the
run order is reproducible from the response alone.

## 3. Add the measured response

Once the runs are done, attach the measured yields back onto the design with
`POST /v1/designs/responses`, which pairs the values to runs **by position** (and
length-checks them) — the wire-format guard against the most damaging mistake in the
process, a reading paired with the wrong run. The values below are the same seeded,
synthetic-but-realistic surface WORKFLOW.md uses, given here in run order; in a real study
this is simply your column of measurements.

```bash
jq -n --slurpfile d design.json '{
  design: $d[0],
  responses: {"yield_pct": [
    71.244, 77.168, 55.6,   63.252, 76.439, 76.958, 74.102, 76.747, 67.487, 76.818,
    77.704, 57.622, 46.053, 56.902, 54.374, 64.313, 77.795, 77.233, 78.703
  ]}
}' | curl -s -X POST http://localhost:8000/v1/designs/responses \
       -H 'Content-Type: application/json' -d @- \
  | jq '.design' > measured.json
```

`measured.json` is now a design document whose runs each carry a `yield_pct` column,
ready for analysis. (Every analysis endpoint refers to the response by the *name* of its
column — `"yield_pct"` — never by a position or a handle.)

## 4. Fit a quadratic model

A central composite design is built for a *quadratic* model — it captures each factor's own
effect, how pairs of factors combine, and the curvature that bends the response toward a
peak. `POST /v1/analysis/fit` re-fits from the design and response every time (OLS is
milliseconds; no fit handle crosses the wire).

```bash
jq -n --slurpfile d measured.json \
  '{design: $d[0], response: "yield_pct", model: "quadratic"}' \
  | curl -s -X POST http://localhost:8000/v1/analysis/fit \
      -H 'Content-Type: application/json' -d @- \
  | jq '{r_squared, adjusted_r2,
         terms: [.terms[] | {term, coefficient, effect, std_error, t, p}]}'
```

```json
{
  "r_squared": 0.998,
  "adjusted_r2": 0.995,
  "terms": [
    {"term": "Intercept",            "coefficient": 77.50, "effect": 77.50, "std_error": 0.26, "t": 293.04, "p": 0.00},
    {"term": "temperature",          "coefficient": 7.39,  "effect": 14.78, "std_error": 0.23, "t": 32.35,  "p": 0.00},
    {"term": "time",                 "coefficient": 5.03,  "effect": 10.05, "std_error": 0.23, "t": 22.00,  "p": 0.00},
    {"term": "catalyst",             "coefficient": 2.93,  "effect": 5.86,  "std_error": 0.23, "t": 12.83,  "p": 0.00},
    {"term": "temperature:time",     "coefficient": 2.53,  "effect": 5.06,  "std_error": 0.26, "t": 9.91,   "p": 0.00},
    {"term": "temperature:catalyst", "coefficient": -0.22, "effect": -0.45, "std_error": 0.26, "t": -0.87,  "p": 0.41},
    {"term": "time:catalyst",        "coefficient": -1.26, "effect": -2.53, "std_error": 0.26, "t": -4.95,  "p": 0.00},
    {"term": "temperature^2",        "coefficient": -7.24, "effect": -14.47, "std_error": 0.44, "t": -16.56, "p": 0.00},
    {"term": "time^2",               "coefficient": -5.61, "effect": -11.21, "std_error": 0.44, "t": -12.83, "p": 0.00},
    {"term": "catalyst^2",           "coefficient": -3.76, "effect": -7.53, "std_error": 0.44, "t": -8.61,  "p": 0.00}
  ]
}
```

The `effect` column reads first: it is the change in yield as a factor sweeps from the low
end of its range to the high end. Temperature's +14.8 says 45 °C → 75 °C lifts yield by ~15
percentage points — the biggest lever here. R² and adjusted R² both sit near 1, the quick
signal the model fits; the next step confirms that is real.

> **The Pareto and effect plots need no endpoint.** They are direct renderings of this
> `terms` array — sort by `|effect|` for a Pareto chart. A frontend draws them from exactly
> the JSON above; the service ships the numbers, not the picture.

## 5. Check whether the model is usable

Three checks, two endpoints. `POST /v1/analysis/anova` returns the term-by-term ANOVA, the
PRESS-based **predicted R²** (the one that guards against a model that only describes the
data it saw), and the **lack-of-fit** test (which weighs the model's misses against the
scatter in the repeated center points). `POST /v1/analysis/diagnostics` returns the
**VIFs** (near 1 means the factors were varied independently, so their effects come through
cleanly).

```bash
jq -n --slurpfile d measured.json \
  '{design: $d[0], response: "yield_pct", model: "quadratic"}' \
  | curl -s -X POST http://localhost:8000/v1/analysis/anova \
      -H 'Content-Type: application/json' -d @- \
  | jq '{predicted_r2, press, lack_of_fit,
         rows: [.rows[] | {term, ss, df, ms, f, p}]}'
```

```json
{
  "predicted_r2": 0.985,
  "press": 28.042,
  "lack_of_fit": {"ss_lof": 1.849, "df_lof": 5, "ss_pe": 2.849, "df_pe": 4, "f": 0.519, "p": 0.755},
  "rows": [
    {"term": "temperature",          "ss": 546.343, "df": 1, "ms": 546.343, "f": 1046.764, "p": 0.000},
    {"term": "time",                 "ss": 252.657, "df": 1, "ms": 252.657, "f": 484.077,  "p": 0.000},
    {"term": "catalyst",             "ss": 85.913,  "df": 1, "ms": 85.913,  "f": 164.606,  "p": 0.000},
    {"term": "temperature:time",     "ss": 51.248,  "df": 1, "ms": 51.248,  "f": 98.188,   "p": 0.000},
    {"term": "temperature:catalyst", "ss": 0.397,   "df": 1, "ms": 0.397,   "f": 0.761,    "p": 0.406},
    {"term": "time:catalyst",        "ss": 12.771,  "df": 1, "ms": 12.771,  "f": 24.469,   "p": 0.001},
    {"term": "temperature^2",        "ss": 758.024, "df": 1, "ms": 758.024, "f": 1452.333, "p": 0.000},
    {"term": "time^2",               "ss": 153.972, "df": 1, "ms": 153.972, "f": 295.002,  "p": 0.000},
    {"term": "catalyst^2",           "ss": 38.710,  "df": 1, "ms": 38.710,  "f": 74.167,   "p": 0.000},
    {"term": "Residual",             "ss": 4.697,   "df": 9, "ms": 0.522,   "f": null,     "p": null},
    {"term": "Total",                "ss": 1904.733, "df": 18, "ms": null,  "f": null,     "p": null}
  ]
}
```

```bash
jq -n --slurpfile d measured.json \
  '{design: $d[0], model: {order: 2, interactions: true}}' \
  | curl -s -X POST http://localhost:8000/v1/analysis/diagnostics \
      -H 'Content-Type: application/json' -d @- \
  | jq '.vif'
```

```json
{
  "temperature": 1.00, "time": 1.00, "catalyst": 1.00,
  "temperature:time": 1.00, "temperature:catalyst": 1.00, "time:catalyst": 1.00,
  "temperature^2": 1.73, "time^2": 1.73, "catalyst^2": 1.73
}
```

All three checks pass. Predicted R² of 0.985 says the model will predict, not just describe.
Every ANOVA term except `temperature:catalyst` stands clearly out of the noise, and the
lack-of-fit p-value of **0.755** says the model's misses are ordinary scatter, not a missing
term. No VIF rises above 1.73 — the payoff of a design that varied the factors independently.

> **The diagnostic plots** (predicted-vs-actual, residuals-vs-fitted) render from `fit`'s
> `fitted`/`residuals` arrays; there is no plot for them to fetch — the numbers are already
> in the `fit` response from step 4.

## 6. Choose the operating point

With a model you trust, ask it the question the study was run to answer.
`POST /v1/optimize/stationary-point` finds the exact top of the fitted surface and classifies
it (a genuine `maximum` here, from the all-negative eigenvalues of the quadratic form).
`POST /v1/optimize/optimum` finds the best setting that stays **inside the tested ranges** —
the answer to trust, since the model is only anchored by data within that box.

```bash
jq -n --slurpfile d measured.json \
  '{design: $d[0], response: "yield_pct", model: "quadratic"}' \
  | curl -s -X POST http://localhost:8000/v1/optimize/stationary-point \
      -H 'Content-Type: application/json' -d @- \
  | jq '{kind, natural, coded, response, eigenvalues}'
```

```json
{
  "kind": "maximum",
  "natural": {"temperature": 69.05, "time": 51.06, "catalyst": 1.779},
  "coded": [0.603, 0.553, 0.279],
  "response": 81.53,
  "eigenvalues": [-7.94, -5.18, -3.49]
}
```

```bash
jq -n --slurpfile d measured.json \
  '{design: $d[0], response: "yield_pct", model: "quadratic", maximize: true}' \
  | curl -s -X POST http://localhost:8000/v1/optimize/optimum \
      -H 'Content-Type: application/json' -d @- \
  | jq '{natural, response, at_bound}'
```

```json
{
  "natural": {"temperature": 69.05, "time": 51.06, "catalyst": 1.779},
  "response": 81.53,
  "at_bound": false
}
```

The peak lands comfortably inside the tested ranges (`at_bound: false`), so both agree:
about **69 °C, 51 minutes, 1.8 g/L catalyst, for a predicted 81.5% yield**. When the peak
instead falls outside the box, `/optimum` stops at the edge of the tested region and reports
`at_bound: true` — it will not recommend a setting the data cannot vouch for. To cap the
search yourself, add a natural-units `bounds` object, e.g.
`"bounds": {"temperature": [45, 70]}`.

> **The contour map** comes from `POST /v1/plot-data/surface` — post the same
> `{design, response, model}` plus `{"x": "temperature", "y": "time", "fixed": {"catalyst": 1.779}}`
> and it returns the `x`/`y`/`z` meshes for a filled-contour renderer, sliced at the best
> catalyst loading.

## 7. Plan the confirmation run

The optimum is still a prediction. Before relying on it, evaluate the model at that setting
with `POST /v1/analysis/predict` — the natural-unit point from `/optimum` goes straight into
`points`:

```bash
jq -n --slurpfile d measured.json '{
  design: $d[0], response: "yield_pct", model: "quadratic",
  points: [{"temperature": 69.05, "time": 51.06, "catalyst": 1.779}]
}' | curl -s -X POST http://localhost:8000/v1/analysis/predict \
       -H 'Content-Type: application/json' -d @- \
  | jq '.predictions'
```

```json
[81.53]
```

The single number, 81.53, is the model's best guess. Run the confirmation point once or
twice; if the measured yield lands near it, the study has delivered an operating point you
can stand behind. If it comes in well below, the model has been pushed past where it holds —
add a handful of runs around this region (`POST /v1/designs/augment` grows the existing
design), re-fit, and re-confirm before committing.

> **Prediction intervals.** WORKFLOW.md closes by turning that point estimate into a 95%
> *prediction interval* (~79.8–83.3%) — the band a single fresh run should fall inside, and
> the right yardstick for a confirmation run. The v1 `/predict` endpoint returns the point
> estimate only; the library's `interval="prediction"` is not yet surfaced on the wire. It
> is a natural additive field on `PredictResponse` — see the deferred items in
> [WEBSERVICE_API.md](WEBSERVICE_API.md#open-questions). Until then, compute the interval
> with the in-process library ([WORKFLOW.md](WORKFLOW.md) step 7) when you need the band.

## Where this differs from the in-process workflow

Same study, same numbers — the differences are all in the plumbing:

- **State lives in the design document, not an object.** You carry `design.json` /
  `measured.json` between calls; the service is stateless and re-fits from inputs each time.
- **The response is a named column.** `yield_pct` is attached to the runs and referred to by
  name everywhere, so a reading can never silently misalign with a run.
- **Randomization is its own step**, and the seed it used is echoed in `meta.random_seed`,
  so any response is reproducible on its own.
- **Plots are data, not images.** Pareto/effect and diagnostic plots render from the `fit`
  response; the contour comes from `/v1/plot-data/surface`. No matplotlib in the service.
- **Prediction intervals** are not yet on the wire (see step 7).

For the full endpoint catalogue see [WEBSERVICE_API.md](WEBSERVICE_API.md); for a single
runnable example of each, [WEBSERVICE_EXAMPLES.md](WEBSERVICE_EXAMPLES.md).
