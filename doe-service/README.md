# doe-service

A stateless HTTP API over the [`doe`](../README.md) library — a uv workspace member of
the `doe` repository, depending on the in-tree `doe` package.

- API contract: [`docs/WEBSERVICE_API.md`](../docs/WEBSERVICE_API.md)
- Architecture and rationale: [`docs/WEBSERVICE.md`](../docs/WEBSERVICE.md)

The dependency points one way only: `doe_service` imports `doe`, never the reverse —
`doe` stays on the scipy stack, and this package can be split into its own repository
later without surgery.

## Development

All commands from this directory:

```bash
uv run --extra dev pytest                              # tests
uv run --extra dev mypy                                # type-check (strict)
uv run --extra dev uvicorn --factory doe_service.main:create_app --reload   # dev server
```

Linting is repo-wide from the repository root: `uv run --extra dev ruff check .`.

## Quickstart

With the dev server running (`uv run --extra dev uvicorn --factory
doe_service.main:create_app --reload`, default `http://127.0.0.1:8000`), generate a
central-composite design and fit a quadratic model to it. This uses
[`jq`](https://jqlang.org/) to reshape the JSON between steps; swap in `python3 -c
"import json,sys; ..."` if you don't have it installed.

```bash
# 1. Generate a 2-factor central-composite design.
curl -s -X POST http://127.0.0.1:8000/v1/designs/central-composite \
  -H 'Content-Type: application/json' \
  -d '{
    "factors": [
      {"type": "continuous", "name": "temp", "low": 20, "high": 80, "units": "C"},
      {"type": "continuous", "name": "time", "low": 0, "high": 10}
    ],
    "alpha": "faced",
    "center": 3
  }' | jq '.design' > design.json

# 2. Attach a synthetic "yield" reading to each run (in practice: your real data).
jq '.runs |= (to_entries | map(.value + {yield: (70 + (.key * 0.6))}))' \
  design.json > design_with_yield.json

# 3. Fit a quadratic model and read back the terms.
jq -n --slurpfile design design_with_yield.json \
  '{design: $design[0], response: "yield", model: "quadratic"}' \
  | curl -s -X POST http://127.0.0.1:8000/v1/analysis/fit \
      -H 'Content-Type: application/json' -d @- \
  | jq '{r_squared, terms: [.terms[] | {term, coefficient}]}'
```

```json
{
  "r_squared": 0.977352472089314,
  "terms": [
    { "term": "Intercept", "coefficient": 75.46315789473684 },
    { "term": "temp", "coefficient": 0.5 },
    { "term": "time", "coefficient": 0.2999999999999989 },
    { "term": "temp:time", "coefficient": 9.43689570931383e-16 },
    { "term": "temp^2", "coefficient": -2.8578947368421037 },
    { "term": "time^2", "coefficient": -1.657894736842108 }
  ]
}
```

See [`docs/WEBSERVICE_API.md`](../docs/WEBSERVICE_API.md) for the full endpoint table
(design generation/operations, analysis, optimization, plot data), the error envelope,
and the deployment-configurable request limits; every worked example there is a passing
contract test under `tests/contract/`.
