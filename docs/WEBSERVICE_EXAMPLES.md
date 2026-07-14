# doe-service — API examples

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: cd doe-service && uv run --extra dev python scripts/build_api_examples.py
     Every request/response pair below is captured live from the running service. -->

A worked, runnable example for every `doe-service` endpoint. This is the cookbook
companion to [WEBSERVICE_API.md](WEBSERVICE_API.md) (the full field-by-field contract)
and [WEBSERVICE.md](WEBSERVICE.md) (architecture). For these calls chained into one
end-to-end study (factors → design → fit → operating point) over HTTP, see
[WEBSERVICE_WORKFLOW.md](WEBSERVICE_WORKFLOW.md); the `doe` library walkthroughs in
[WORKFLOW.md](WORKFLOW.md) and the [vignettes](VIGNETTES.md) show the same computations
in-process.

Every pair below is **real** — captured by POSTing through the actual FastAPI app
(`scripts/build_api_examples.py`), so it is exactly what the service returns today.

## Running the examples

Start the dev server from the `doe-service` directory:

```bash
uv run --extra dev uvicorn --factory doe_service.main:create_app --reload
```

It listens on `http://127.0.0.1:8000` by default (the examples below write
`http://localhost:8000`; use whichever host you started). Every compute endpoint is a
`POST` with a JSON body; `GET /v1/health` and the auto-generated `/openapi.json`
(interactive docs at `/docs`) are the only non-`POST` routes.

Conventions used throughout — full detail in [WEBSERVICE_API.md](WEBSERVICE_API.md):

- **Natural units on the wire.** Runs, bounds, `fixed` values, and prediction points
  are natural-unit values; only explicit `region` candidate arrays and the `coded`
  echoes in optimization results are coded.
- **The design document is the wire format.** Anywhere a request carries a `design`, it
  is a `Design.to_dict()` document ([SERIALIZATION.md](SERIALIZATION.md)); anywhere a
  response returns one, it came straight from `Design.to_dict()`. Round-trip a response
  design back into the next request unchanged.
- **Responses are named columns.** Analysis endpoints take `response` as the *name* of a
  run column. Attach readings with [`POST /v1/designs/responses`](#attach-responses) (or
  write the column into the document yourself — the validator accepts extra run columns).
- **Warnings are data.** Responses carry a `"warnings"` array (often empty); it never
  changes the HTTP status.

> **Note on abbreviation.** To keep this page readable, long arrays (runs, fitted
> values, residuals, plot meshes) are shortened to the first few entries followed by a
> `"… N more"` marker. The live service returns the arrays in full.

## Health check

A cheap liveness probe — the one `GET` compute-adjacent route.

### Service health

```bash
curl -s -X GET http://localhost:8000/v1/health \
  -H 'Content-Type: application/json'
```

`200`

```json
{
  "status": "ok",
  "doe_version": "0.1.0"
}
```

## Design generation

One endpoint per generator. Each takes `{factors: [...], ...params}` and returns `{design, warnings}`, where `design` is a full design document you can feed straight into analysis or an operation.

### Full factorial

Every combination of factor levels — here the 2-level, 2-factor 2² design (add `"levels": 3` or a per-factor list for more levels).

```bash
curl -s -X POST http://localhost:8000/v1/designs/full-factorial \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "levels": 2
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "warnings": []
}
```

### Fractional factorial

A 2^(4−1) half-fraction from the defining relation `D=ABC`: four factors in eight runs.

```bash
curl -s -X POST http://localhost:8000/v1/designs/fractional-factorial \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "A",
      "low": -1,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "B",
      "low": -1,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "C",
      "low": -1,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "D",
      "low": -1,
      "high": 1
    }
  ],
  "generators": [
    "D=ABC"
  ]
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "fractional_factorial_4-1",
    "factors": [
      {
        "type": "continuous",
        "name": "A",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "B",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "C",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "D",
        "low": -1.0,
        "high": 1.0,
        "units": null
      }
    ],
    "runs": [
      {
        "A": -1.0,
        "B": -1.0,
        "C": -1.0,
        "D": -1.0
      },
      {
        "A": -1.0,
        "B": -1.0,
        "C": 1.0,
        "D": 1.0
      },
      {
        "A": -1.0,
        "B": 1.0,
        "C": -1.0,
        "D": 1.0
      },
      {
        "A": -1.0,
        "B": 1.0,
        "C": 1.0,
        "D": -1.0
      },
      "… 4 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "fractional_factorial",
        "parameters": {
          "generators": [
            "D=ABC"
          ]
        }
      }
    }
  },
  "warnings": []
}
```

### Plackett–Burman

A saturated, orthogonal main-effect screening design; run count comes from the smallest constructible Hadamard order that fits the factors.

```bash
curl -s -X POST http://localhost:8000/v1/designs/plackett-burman \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "a",
      "low": 0,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "b",
      "low": 0,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "c",
      "low": 0,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "d",
      "low": 0,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "e",
      "low": 0,
      "high": 1
    }
  ]
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "plackett_burman_8",
    "factors": [
      {
        "type": "continuous",
        "name": "a",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "b",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "c",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "d",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      "… 1 more"
    ],
    "runs": [
      {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "d": 1.0,
        "e": 1.0
      },
      {
        "a": 0.0,
        "b": 1.0,
        "c": 0.0,
        "d": 1.0,
        "e": 0.0
      },
      {
        "a": 1.0,
        "b": 0.0,
        "c": 0.0,
        "d": 1.0,
        "e": 1.0
      },
      {
        "a": 0.0,
        "b": 0.0,
        "c": 1.0,
        "d": 1.0,
        "e": 0.0
      },
      "… 4 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "plackett_burman",
        "parameters": {}
      }
    }
  },
  "warnings": []
}
```

### Definitive screening design

A conference-matrix DSD: `2k+1` runs, main effects orthogonal to all second-order terms. Include a categorical factor and it takes the Jones–Nachtsheim categorical DSD-augment path automatically (no new endpoint).

```bash
curl -s -X POST http://localhost:8000/v1/designs/definitive-screening \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10
    },
    {
      "type": "continuous",
      "name": "conc",
      "low": 1,
      "high": 5
    }
  ]
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "definitive_screening_k3",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "conc",
        "low": 1.0,
        "high": 5.0,
        "units": null
      }
    ],
    "runs": [
      {
        "temp": 50.0,
        "time": 10.0,
        "conc": 5.0
      },
      {
        "temp": 80.0,
        "time": 5.0,
        "conc": 1.0
      },
      {
        "temp": 80.0,
        "time": 10.0,
        "conc": 3.0
      },
      {
        "temp": 80.0,
        "time": 0.0,
        "conc": 5.0
      },
      "… 5 more"
    ],
    "point_types": [
      "dsd",
      "dsd",
      "dsd",
      "dsd",
      "… 5 more"
    ],
    "meta": {
      "generator": {
        "name": "definitive_screening",
        "parameters": {
          "extra_center_runs": 0,
          "fake_factors": null
        }
      },
      "fake_factors": 1
    }
  },
  "warnings": []
}
```

### Central composite (RSM)

A rotatable central-composite design: factorial core + axial (`alpha`) points + center replicates, ready for a quadratic fit.

```bash
curl -s -X POST http://localhost:8000/v1/designs/central-composite \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "alpha": "rotatable",
  "center": 5
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      },
      "… 9 more"
    ],
    "point_types": [
      "factorial",
      "factorial",
      "factorial",
      "factorial",
      "… 9 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "rotatable",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.4142135623730951,
      "axial_extrapolates": true
    }
  },
  "warnings": []
}
```

### Box–Behnken (RSM)

A 3-level second-order design with no corner runs — useful when the extreme corner combinations are infeasible. Needs ≥3 continuous factors.

```bash
curl -s -X POST http://localhost:8000/v1/designs/box-behnken \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10
    },
    {
      "type": "continuous",
      "name": "conc",
      "low": 1,
      "high": 5
    }
  ],
  "center": 3
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "box_behnken_k3",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "conc",
        "low": 1.0,
        "high": 5.0,
        "units": null
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0,
        "conc": 3.0
      },
      {
        "temp": 20.0,
        "time": 10.0,
        "conc": 3.0
      },
      {
        "temp": 80.0,
        "time": 0.0,
        "conc": 3.0
      },
      {
        "temp": 80.0,
        "time": 10.0,
        "conc": 3.0
      },
      "… 11 more"
    ],
    "point_types": [
      "edge",
      "edge",
      "edge",
      "edge",
      "… 11 more"
    ],
    "meta": {
      "generator": {
        "name": "box_behnken",
        "parameters": {
          "center": 3
        }
      }
    }
  },
  "warnings": []
}
```

### Space-filling (Latin hypercube / Sobol / Halton)

Coverage-oriented sampling for computer experiments and surrogate modelling. `sampler` picks the engine; `sobol` requires power-of-two `n_runs` (it names the nearest valid sizes otherwise). Seeded for reproducibility.

```bash
curl -s -X POST http://localhost:8000/v1/designs/space-filling \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "sampler": "lhs",
  "n_runs": 8,
  "seed": 7
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 77.45023479045886,
        "time": 6.18719537898317
      },
      {
        "temp": 46.51494698997086,
        "time": 0.5928471125925743
      },
      {
        "temp": 59.42906563549343,
        "time": 9.198618445881529
      },
      {
        "temp": 51.47353381406448,
        "time": 3.776053139823814
      },
      "… 4 more"
    ],
    "point_types": null,
    "meta": {
      "sampler": "lhs",
      "criterion": "maximin",
      "seed": 7
    }
  },
  "warnings": []
}
```

### Simplex-lattice (mixture)

A `{k, m}` lattice over the mixture simplex — every blend whose proportions are multiples of `1/degree`. Needs the full simplex (unconstrained components); use `extreme-vertices` when components are bounded.

```bash
curl -s -X POST http://localhost:8000/v1/designs/simplex-lattice \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "mixture",
      "name": "polymer"
    },
    {
      "type": "mixture",
      "name": "solvent"
    },
    {
      "type": "mixture",
      "name": "additive"
    }
  ],
  "degree": 2
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "simplex_lattice_3_2",
    "factors": [
      {
        "type": "mixture",
        "name": "polymer",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "mixture",
        "name": "solvent",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "mixture",
        "name": "additive",
        "low": 0.0,
        "high": 1.0,
        "units": null
      }
    ],
    "runs": [
      {
        "polymer": 0.0,
        "solvent": 0.0,
        "additive": 1.0
      },
      {
        "polymer": 0.0,
        "solvent": 0.5,
        "additive": 0.5
      },
      {
        "polymer": 0.0,
        "solvent": 1.0,
        "additive": 0.0
      },
      {
        "polymer": 0.5,
        "solvent": 0.0,
        "additive": 0.5
      },
      "… 2 more"
    ],
    "point_types": [
      "vertex",
      "edge-centroid",
      "vertex",
      "edge-centroid",
      "… 2 more"
    ],
    "meta": {
      "generator": "simplex_lattice",
      "degree": 2
    }
  },
  "warnings": []
}
```

### Simplex-centroid (mixture)

The `2^k − 1` subset centroids of the (full) simplex.

```bash
curl -s -X POST http://localhost:8000/v1/designs/simplex-centroid \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "mixture",
      "name": "polymer"
    },
    {
      "type": "mixture",
      "name": "solvent"
    },
    {
      "type": "mixture",
      "name": "additive"
    }
  ]
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "simplex_centroid_3",
    "factors": [
      {
        "type": "mixture",
        "name": "polymer",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "mixture",
        "name": "solvent",
        "low": 0.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "mixture",
        "name": "additive",
        "low": 0.0,
        "high": 1.0,
        "units": null
      }
    ],
    "runs": [
      {
        "polymer": 1.0,
        "solvent": 0.0,
        "additive": 0.0
      },
      {
        "polymer": 0.0,
        "solvent": 1.0,
        "additive": 0.0
      },
      {
        "polymer": 0.0,
        "solvent": 0.0,
        "additive": 1.0
      },
      {
        "polymer": 0.5,
        "solvent": 0.5,
        "additive": 0.0
      },
      "… 3 more"
    ],
    "point_types": [
      "vertex",
      "vertex",
      "vertex",
      "edge-centroid",
      "… 3 more"
    ],
    "meta": {
      "generator": "simplex_centroid"
    }
  },
  "warnings": []
}
```

### Extreme vertices (mixture)

The vertices of a bound-constrained simplex (McLean–Anderson XVERT) plus the centroid — the design to reach for when component bounds carve a polytope out of the full simplex.

```bash
curl -s -X POST http://localhost:8000/v1/designs/extreme-vertices \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "mixture",
      "name": "polymer",
      "low": 0.2,
      "high": 0.6
    },
    {
      "type": "mixture",
      "name": "solvent",
      "low": 0.2,
      "high": 0.6
    },
    {
      "type": "mixture",
      "name": "additive",
      "low": 0.0,
      "high": 0.4
    }
  ],
  "include_centroid": true
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "extreme_vertices_3",
    "factors": [
      {
        "type": "mixture",
        "name": "polymer",
        "low": 0.2,
        "high": 0.6,
        "units": null
      },
      {
        "type": "mixture",
        "name": "solvent",
        "low": 0.2,
        "high": 0.6,
        "units": null
      },
      {
        "type": "mixture",
        "name": "additive",
        "low": 0.0,
        "high": 0.4,
        "units": null
      }
    ],
    "runs": [
      {
        "polymer": 0.2,
        "solvent": 0.4,
        "additive": 0.4
      },
      {
        "polymer": 0.2,
        "solvent": 0.6,
        "additive": 0.2
      },
      {
        "polymer": 0.4,
        "solvent": 0.2,
        "additive": 0.4
      },
      {
        "polymer": 0.4,
        "solvent": 0.6,
        "additive": 0.0
      },
      "… 3 more"
    ],
    "point_types": [
      "vertex",
      "vertex",
      "vertex",
      "vertex",
      "… 3 more"
    ],
    "meta": {
      "generator": "extreme_vertices",
      "include_centroid": true,
      "n_vertices": 6
    }
  },
  "warnings": []
}
```

## Computer-generated optimal designs

The coordinate-exchange engine for when no named recipe fits (odd run budgets, mixed factor types, constrained regions). These return a `search` report alongside the design.

### D-optimal design

Build a 12-run D-optimal design for a quadratic model by coordinate exchange. `criterion` is `"D"` (default) or `"I"`; `region` (coded candidate rows) is optional — omitted, it defaults to a `candidate_grid`.

```bash
curl -s -X POST http://localhost:8000/v1/designs/optimal \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "n_runs": 12,
  "model": "quadratic",
  "criterion": "D",
  "n_restarts": 10,
  "seed": 42
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "d_optimal_quadratic",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 50.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      "… 8 more"
    ],
    "point_types": null,
    "meta": {
      "criterion": "D",
      "score": 10.319562839756308,
      "d_efficiency": 0.4653434658439953,
      "n_restarts": 10,
      "seed": 42,
      "model": "quadratic",
      "order": 2,
      "interactions": true,
      "converged": true
    }
  },
  "search": {
    "criterion": "D",
    "score": 10.319562839756308,
    "d_efficiency": 0.4653434658439953,
    "n_restarts": 10,
    "converged": true
  },
  "warnings": []
}
```

### Candidate set

The discrete candidate points the optimal engine searches over — request them so a client can filter to a feasible region (WORKFLOW6) before passing them back as `region`. `kind` is `"grid"` for box factors, `"mixture"` for a simplex.

```bash
curl -s -X POST http://localhost:8000/v1/designs/candidates \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "levels": 3
}'
```

`200`

```json
{
  "points": [
    [
      -1.0,
      -1.0
    ],
    [
      -1.0,
      0.0
    ],
    [
      -1.0,
      1.0
    ],
    [
      0.0,
      -1.0
    ],
    "… 5 more"
  ],
  "kind": "grid"
}
```

## Split-plot & blocking (Phase 5)

Restricted-randomization generators. Same `{factors, ...}` → `{design, warnings}` shape; the block-carrying generators add a reserved `block` categorical factor, and `split-plot` returns a `whole_plots` array (one plot id per run) that `fit-gls` consumes.

### Split-plot design

Cross a whole-plot design over the hard-to-change factors with a sub-plot design run inside each plot. Flag the hard-to-change factor with `"hard_to_change": true`.

```bash
curl -s -X POST http://localhost:8000/v1/designs/split-plot \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 100,
      "high": 200,
      "hard_to_change": true
    },
    {
      "type": "continuous",
      "name": "conc",
      "low": 1,
      "high": 5
    }
  ],
  "seed": 1
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "split_plot_1wp_1sp",
    "factors": [
      {
        "hard_to_change": true,
        "type": "continuous",
        "name": "temp",
        "low": 100.0,
        "high": 200.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "conc",
        "low": 1.0,
        "high": 5.0,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 100.0,
        "conc": 1.0
      },
      {
        "std_order": 1,
        "temp": 100.0,
        "conc": 5.0
      },
      {
        "std_order": 3,
        "temp": 200.0,
        "conc": 5.0
      },
      {
        "std_order": 2,
        "temp": 200.0,
        "conc": 1.0
      }
    ],
    "point_types": null,
    "whole_plots": [
      0,
      0,
      1,
      1
    ],
    "meta": {
      "generator": {
        "name": "split_plot",
        "parameters": {
          "whole_plot_design": "full",
          "sub_plot_design": "full",
          "n_whole_plot_reps": 1,
          "seed": 1
        }
      },
      "randomized": true,
      "random_seed": 1
    }
  },
  "warnings": []
}
```

### Randomized complete block

Each of `n_blocks` blocks sees every treatment once. Give either an explicit `factors` list or `n_treatments`.

```bash
curl -s -X POST http://localhost:8000/v1/designs/randomized-complete-block \
  -H 'Content-Type: application/json' \
  -d '{
  "n_treatments": 3,
  "n_blocks": 4,
  "seed": 3
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "rcb_3x4",
    "factors": [
      {
        "type": "categorical",
        "name": "treatment",
        "levels": [
          "T1",
          "T2",
          "T3"
        ],
        "units": null
      },
      {
        "type": "categorical",
        "name": "block",
        "levels": [
          "B1",
          "B2",
          "B3",
          "B4"
        ],
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 2,
        "treatment": "T3",
        "block": "B1"
      },
      {
        "std_order": 1,
        "treatment": "T2",
        "block": "B1"
      },
      {
        "std_order": 0,
        "treatment": "T1",
        "block": "B1"
      },
      {
        "std_order": 3,
        "treatment": "T1",
        "block": "B2"
      },
      "… 8 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "randomized_complete_block",
        "parameters": {
          "treatments": 3,
          "n_blocks": 4,
          "seed": 3
        }
      },
      "randomized": true,
      "random_seed": 3
    }
  },
  "warnings": []
}
```

### Latin square

A `t × t` square controlling two nuisance directions (row/column blocks) at once.

```bash
curl -s -X POST http://localhost:8000/v1/designs/latin-square \
  -H 'Content-Type: application/json' \
  -d '{
  "treatments": 4,
  "seed": 5
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "latin_square_4",
    "factors": [
      {
        "type": "categorical",
        "name": "row",
        "levels": [
          "R1",
          "R2",
          "R3",
          "R4"
        ],
        "units": null
      },
      {
        "type": "categorical",
        "name": "column",
        "levels": [
          "C1",
          "C2",
          "C3",
          "C4"
        ],
        "units": null
      },
      {
        "type": "categorical",
        "name": "treatment",
        "levels": [
          "T1",
          "T2",
          "T3",
          "T4"
        ],
        "units": null
      }
    ],
    "runs": [
      {
        "row": "R1",
        "column": "C1",
        "treatment": "T2"
      },
      {
        "row": "R1",
        "column": "C2",
        "treatment": "T1"
      },
      {
        "row": "R1",
        "column": "C3",
        "treatment": "T3"
      },
      {
        "row": "R1",
        "column": "C4",
        "treatment": "T4"
      },
      "… 12 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "latin_square",
        "parameters": {
          "treatments": 4,
          "seed": 5
        }
      }
    }
  },
  "warnings": []
}
```

### Blocked factorial

A 2³ factorial split into blocks by confounding the `ABC` contrast; the full confounded set is recorded in `meta.confounded_with_blocks`.

```bash
curl -s -X POST http://localhost:8000/v1/designs/blocked-factorial \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "A",
      "low": -1,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "B",
      "low": -1,
      "high": 1
    },
    {
      "type": "continuous",
      "name": "C",
      "low": -1,
      "high": 1
    }
  ],
  "block_generators": [
    "ABC"
  ],
  "seed": 2
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "blocked_factorial_3_2blocks",
    "factors": [
      {
        "type": "continuous",
        "name": "A",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "B",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "C",
        "low": -1.0,
        "high": 1.0,
        "units": null
      },
      {
        "type": "categorical",
        "name": "block",
        "levels": [
          "B1",
          "B2"
        ],
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 6,
        "A": 1.0,
        "B": 1.0,
        "C": -1.0,
        "block": "B1"
      },
      {
        "std_order": 5,
        "A": 1.0,
        "B": -1.0,
        "C": 1.0,
        "block": "B1"
      },
      {
        "std_order": 0,
        "A": -1.0,
        "B": -1.0,
        "C": -1.0,
        "block": "B1"
      },
      {
        "std_order": 3,
        "A": -1.0,
        "B": 1.0,
        "C": 1.0,
        "block": "B1"
      },
      "… 4 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "blocked_factorial",
        "parameters": {
          "block_generators": [
            "ABC"
          ],
          "seed": 2
        }
      },
      "confounded_with_blocks": [
        "ABC"
      ],
      "randomized": true,
      "random_seed": 2
    }
  },
  "warnings": []
}
```

## Design operations

Pure transformations: a design document in, a design document (or a validation verdict) out.

### Validate a design document

The one endpoint that returns `200` for an *invalid* design too — the verdict is the payload. `check_ranges` additionally flags runs outside a factor's declared range.

```bash
curl -s -X POST http://localhost:8000/v1/designs/validate \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "check_ranges": false
}'
```

`200`

```json
{
  "valid": true,
  "errors": []
}
```

### Randomize run order

Shuffle run order (respecting split-plot structure when present). The seed used is echoed in `meta` so the order is reproducible.

```bash
curl -s -X POST http://localhost:8000/v1/designs/randomize \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "seed": 11
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "std_order": 3,
        "temp": 80.0,
        "time": 10.0
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0
      },
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0
      },
      {
        "std_order": 2,
        "temp": 80.0,
        "time": 0.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      },
      "randomized": true,
      "random_seed": 11
    }
  },
  "warnings": []
}
```

### Replicate runs

Duplicate the whole design `n` times (`each: true` instead repeats each run in place).

```bash
curl -s -X POST http://localhost:8000/v1/designs/replicate \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "n": 2,
  "each": false
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      },
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      "… 2 more"
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      },
      "replicates": 2
    }
  },
  "warnings": []
}
```

### Project onto a subset of factors

Drop columns for the factors that didn't survive screening, keeping runs and any responses — the setup for `augment`.

```bash
curl -s -X POST http://localhost:8000/v1/designs/project \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "factors": [
    "temp"
  ]
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      }
    ],
    "runs": [
      {
        "temp": 20.0
      },
      {
        "temp": 20.0
      },
      {
        "temp": 80.0
      },
      {
        "temp": 80.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "warnings": []
}
```

### Attach responses

Pair measured readings to runs by position (length-checked). The returned document carries the new column, ready for analysis.

```bash
curl -s -X POST http://localhost:8000/v1/designs/responses \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0
      },
      {
        "temp": 20.0,
        "time": 10.0
      },
      {
        "temp": 80.0,
        "time": 0.0
      },
      {
        "temp": 80.0,
        "time": 10.0
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "responses": {
    "yield": [
      63.1,
      71.4,
      68.9,
      74.2
    ]
  }
}'
```

`200`

```json
{
  "design": {
    "schema_version": "1.0",
    "name": "full_factorial_2x2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20.0,
        "high": 80.0,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0.0,
        "high": 10.0,
        "units": "min"
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0,
        "yield": 63.1
      },
      {
        "temp": 20.0,
        "time": 10.0,
        "yield": 71.4
      },
      {
        "temp": 80.0,
        "time": 0.0,
        "yield": 68.9
      },
      {
        "temp": 80.0,
        "time": 10.0,
        "yield": 74.2
      }
    ],
    "point_types": null,
    "meta": {
      "generator": {
        "name": "full_factorial",
        "parameters": {
          "levels": [
            2,
            2
          ]
        }
      }
    }
  },
  "warnings": []
}
```

## Analysis

Every analysis endpoint takes `{design, response, model}` and re-fits internally — no fit handle crosses the wire. The examples below all use the same golden quadratic-`yield` central-composite design (13 runs, `temp` 20–80 °C, `time` 0–10 min) that the contract tests lock in, so the numbers match [WEBSERVICE_API.md](WEBSERVICE_API.md).

### Fit an OLS model

Least-squares fit with coefficients, effects, standard errors, t/p values and confidence intervals per term, plus `r_squared`/`adjusted_r2` and fitted/residual vectors. Saturated fits return `null` inference columns + a `"saturated_model"` warning; Scheffé fits return `effect: null`.

```bash
curl -s -X POST http://localhost:8000/v1/analysis/fit \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic",
  "confidence": 0.95
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "terms": [
    {
      "term": "Intercept",
      "coefficient": 70.10705699780807,
      "effect": 70.10705699780807,
      "std_error": 0.09453330855157353,
      "t": 741.6122218927787,
      "p": 2.140522787971692e-18,
      "ci_low": 69.88352124382372,
      "ci_high": 70.33059275179242
    },
    {
      "term": "temp",
      "coefficient": 2.133386638852627,
      "effect": 4.266773277705254,
      "std_error": 0.09294440028775322,
      "t": 22.953363863209862,
      "p": 7.551182348598893e-08,
      "ci_low": 1.9136080558824584,
      "ci_high": 2.3531652218227954
    },
    {
      "term": "time",
      "coefficient": 1.3984757624286601,
      "effect": 2.7969515248573202,
      "std_error": 0.09294440028775322,
      "t": 15.046369206741009,
      "p": 1.3759309080919928e-06,
      "ci_low": 1.1786971794584915,
      "ci_high": 1.6182543453988287
    },
    "… 3 more"
  ],
  "r_squared": 0.9965954680087256,
  "adjusted_r2": 0.9941636594435296,
  "dof_resid": 7,
  "mse": 0.051831969269100595,
  "fitted": [
    62.441177194293154,
    69.44067289708941,
    63.038460205546464,
    "… 10 more"
  ],
  "residuals": [
    -0.17911393684052257,
    0.13149848797206687,
    -0.10735811042618337,
    "… 10 more"
  ],
  "model": {
    "order": 2,
    "interactions": true
  },
  "warnings": []
}
```

### ANOVA + lack-of-fit

Sequential (Type I) sums of squares, plus a lack-of-fit split when the design has replicate center points (otherwise `lack_of_fit: null` + a `"no_pure_error"` warning) and PRESS-based predictive R².

```bash
curl -s -X POST http://localhost:8000/v1/analysis/anova \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic"
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "rows": [
    {
      "term": "temp",
      "ss": 27.308031305009465,
      "df": 1.0,
      "ms": 27.308031305009465,
      "f": 526.8569126369086,
      "p": 7.551182348598881e-08
    },
    {
      "term": "time",
      "ss": 11.734406748602616,
      "df": 1.0,
      "ms": 11.734406748602616,
      "f": 226.39322630556566,
      "p": 1.3759309080919602e-06
    },
    {
      "term": "temp:time",
      "ss": 4.838541569740718,
      "df": 1.0,
      "ms": 4.838541569740718,
      "f": 93.350525514862,
      "p": 2.6828524483181086e-05
    },
    {
      "term": "temp^2",
      "ss": 50.5512372374548,
      "df": 1.0,
      "ms": 50.5512372374548,
      "f": 975.2906931821072,
      "p": 8.915429486173878e-09
    },
    {
      "term": "time^2",
      "ss": 11.775784893206403,
      "df": 1.0,
      "ms": 11.775784893206403,
      "f": 227.1915394931075,
      "p": 1.359529310772325e-06
    },
    {
      "term": "Residual",
      "ss": 0.36282378488370415,
      "df": 7.0,
      "ms": 0.051831969269100595,
      "f": null,
      "p": null
    },
    {
      "term": "Total",
      "ss": 106.57082553889848,
      "df": 12.0,
      "ms": null,
      "f": null,
      "p": null
    }
  ],
  "lack_of_fit": {
    "ss_lof": 0.27583370370697446,
    "df_lof": 3,
    "ss_pe": 0.08699008117672971,
    "df_pe": 4,
    "f": 4.227818466591819,
    "p": 0.0987389979990755
  },
  "press": 2.1289671696948753,
  "predicted_r2": 0.9800229832233232,
  "warnings": []
}
```

### Predict at new points

Evaluate the fitted model at natural-unit points — the workhorse for grid searches over constrained regions. Each point must cover every factor.

```bash
curl -s -X POST http://localhost:8000/v1/analysis/predict \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic",
  "points": [
    {
      "temp": 55,
      "time": 4.2
    },
    {
      "temp": 20,
      "time": 10
    }
  ]
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "predictions": [
    70.06864843973118,
    63.038460205546464
  ]
}
```

### Design diagnostics

Judge a design against a model *before* running it — no `response` needed. Returns D/A/G/I efficiencies, the condition number, per-term VIFs, the correlation (alias) matrix, and per-run leverage.

```bash
curl -s -X POST http://localhost:8000/v1/analysis/diagnostics \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "model": {
    "order": 2,
    "interactions": true
  }
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "efficiency": {
    "d": 0.38891980671699955,
    "a": 0.3118745332337566,
    "g": 0.5840559440559441,
    "i": 0.535964035964036
  },
  "condition_number": 3.1715015347471494,
  "vif": {
    "temp": 1.0000000000000002,
    "time": 1.0000000000000002,
    "temp:time": 1.0,
    "temp^2": 1.169761273209549,
    "time^2": 1.169761273209549
  },
  "correlation_matrix": {
    "labels": [
      "temp",
      "time",
      "temp:time",
      "temp^2",
      "time^2"
    ],
    "matrix": [
      [
        0.9999999999999999,
        0.0,
        0.0,
        0.0,
        2.5216315845372088e-17
      ],
      [
        0.0,
        0.9999999999999999,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        0.0,
        1.0,
        0.3809523809523809
      ],
      [
        2.5216315845372088e-17,
        0.0,
        0.0,
        0.3809523809523809,
        1.0
      ]
    ]
  },
  "leverage": [
    0.7902298850574718,
    0.49425287356321856,
    0.7902298850574714,
    0.49425287356321856,
    0.49425287356321845,
    0.7902298850574713,
    "… 7 more"
  ]
}
```

### Coverage metrics

Model-free space-filling quality: `discrepancy` (`method` one of `CD`/`WD`/`MD`/`L2-star`) and `maximin_distance`, both on the coded design rescaled to the unit cube.

```bash
curl -s -X POST http://localhost:8000/v1/analysis/coverage \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "method": "CD"
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "discrepancy": 0.04565253122945401,
  "maximin_distance": 0.0
}
```

### Split-plot fit (REML/GLS)

The split-plot front door: requires a design carrying `whole_plots`, estimates the variance ratio by REML, and returns everything `fit` does plus the whole-plot variance component (`sigma2_wp`) and two-stratum degrees of freedom (`dof_terms`) — the coarser standard errors OLS understates.

```bash
curl -s -X POST http://localhost:8000/v1/analysis/fit-gls \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "split_plot_1wp_1sp",
    "factors": [
      {
        "hard_to_change": true,
        "type": "continuous",
        "name": "temp",
        "low": 100.0,
        "high": 200.0,
        "units": null
      },
      {
        "type": "continuous",
        "name": "conc",
        "low": 1.0,
        "high": 5.0,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 100.0,
        "conc": 1.0,
        "y": 50.0
      },
      {
        "std_order": 1,
        "temp": 100.0,
        "conc": 5.0,
        "y": 51.0
      },
      {
        "std_order": 3,
        "temp": 200.0,
        "conc": 5.0,
        "y": 50.0
      },
      {
        "std_order": 2,
        "temp": 200.0,
        "conc": 1.0,
        "y": 51.0
      },
      {
        "std_order": 5,
        "temp": 100.0,
        "conc": 5.0,
        "y": 54.0
      },
      {
        "std_order": 4,
        "temp": 100.0,
        "conc": 1.0,
        "y": 50.0
      },
      {
        "std_order": 7,
        "temp": 200.0,
        "conc": 5.0,
        "y": 49.0
      },
      {
        "std_order": 6,
        "temp": 200.0,
        "conc": 1.0,
        "y": 50.0
      }
    ],
    "point_types": null,
    "whole_plots": [
      0,
      0,
      1,
      1,
      2,
      2,
      3,
      3
    ],
    "meta": {
      "generator": {
        "name": "split_plot",
        "parameters": {
          "whole_plot_design": "full",
          "sub_plot_design": "full",
          "n_whole_plot_reps": 2,
          "seed": 1
        }
      },
      "randomized": true,
      "random_seed": 1
    }
  },
  "response": "y",
  "model": "linear",
  "confidence": 0.95
}'
```

`200`

```json
{
  "terms": [
    {
      "term": "Intercept",
      "coefficient": 50.62500000000001,
      "effect": 50.62500000000001,
      "std_error": 0.4506939284323682,
      "t": 112.32678500038162,
      "p": 7.924679654808261e-05,
      "ci_low": 48.68582053854897,
      "ci_high": 52.56417946145105
    },
    {
      "term": "temp",
      "coefficient": -0.6250000000000009,
      "effect": -1.2500000000000018,
      "std_error": 0.4506939284323682,
      "t": -1.3867504321034787,
      "p": 0.29985997303284434,
      "ci_low": -2.5641794614510385,
      "ci_high": 1.314179461451037
    },
    {
      "term": "conc",
      "coefficient": 0.375,
      "effect": 0.75,
      "std_error": 0.3749999841914659,
      "t": 1.0000000421560926,
      "p": 0.42264971458448697,
      "ci_low": -1.2384947056374158,
      "ci_high": 1.9884947056374158
    },
    "… 1 more"
  ],
  "r_squared": 0.6535433070866141,
  "adjusted_r2": 0.39370078740157477,
  "dof_resid": 2,
  "mse": 1.1249999051487976,
  "fitted": [
    50.00000000000001,
    52.50000000000001,
    49.50000000000001,
    "… 5 more"
  ],
  "residuals": [
    -7.105427357601002e-15,
    -1.500000000000007,
    0.4999999999999929,
    "… 5 more"
  ],
  "model": {
    "order": 1,
    "interactions": true
  },
  "warnings": [],
  "sigma2_wp": 0.25000011592880367,
  "n_whole_plots": 4,
  "dof_terms": {
    "Intercept": 2,
    "temp": 2,
    "conc": 2,
    "temp:conc": 2
  }
}
```

## Optimization

Read the optimum off a fitted quadratic surface. The first two reuse the golden `yield` design; `desirability` balances three competing responses on one design.

### Stationary point + canonical analysis

Solve `−½ B⁻¹ b` for the surface's stationary point and classify it (`maximum`/`minimum`/`saddle`) via the eigen-decomposition of the quadratic form.

```bash
curl -s -X POST http://localhost:8000/v1/optimize/stationary-point \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic"
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "kind": "maximum",
  "natural": {
    "temp": 62.43573493079263,
    "time": 7.245169319200981
  },
  "coded": [
    0.4145244976930875,
    0.44903386384019617
  ],
  "response": 70.86320899778093,
  "eigenvalues": [
    -3.3961470300336924,
    -1.8377046290019483
  ],
  "eigenvectors": [
    [
      -0.9242522041606367,
      -0.38178248140034543
    ],
    [
      0.38178248140034543,
      -0.9242522041606367
    ]
  ],
  "response_name": "yield",
  "warnings": []
}
```

### Constrained optimum

Multistart L-BFGS-B search over the coded box (or explicit `bounds`); `at_bound` flags when the optimum sits on a constraint.

```bash
curl -s -X POST http://localhost:8000/v1/optimize/optimum \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic",
  "maximize": true
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "natural": {
    "temp": 62.435734951761404,
    "time": 7.245169322626485
  },
  "coded": [
    0.4145244983920467,
    0.44903386452529703
  ],
  "response": 70.86320899778094,
  "maximize": true,
  "at_bound": false,
  "response_name": "yield",
  "warnings": []
}
```

### Desirability (multi-response)

Derringer–Suich geometric-mean desirability across several goals (`max`/`min`/`target`), each re-fit on the shared design. Returns the best operating point plus per-response predictions and individual desirabilities.

```bash
curl -s -X POST http://localhost:8000/v1/optimize/desirability \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 0.0,
        "yield_pct": 52.0,
        "impurity_pct": 0.7,
        "cost": 9.0
      },
      {
        "temp": 20.0,
        "time": 10.0,
        "yield_pct": 62.0,
        "impurity_pct": 1.9000000000000001,
        "cost": 8.0
      },
      {
        "temp": 80.0,
        "time": 0.0,
        "yield_pct": 68.0,
        "impurity_pct": 2.6999999999999997,
        "cost": 12.0
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "factorial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false
    }
  },
  "goals": [
    {
      "response": "yield_pct",
      "model": "quadratic",
      "goal": "max",
      "low": 60,
      "high": 90,
      "weight": 1.0
    },
    {
      "response": "impurity_pct",
      "model": "quadratic",
      "goal": "min",
      "low": 0.5,
      "high": 3.0
    },
    {
      "response": "cost",
      "model": "linear",
      "goal": "target",
      "low": 8,
      "target": 10,
      "high": 12
    }
  ]
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "natural": {
    "temp": 49.1716649105463,
    "time": 4.585832455284189
  },
  "coded": [
    -0.02761116964845672,
    -0.08283350894316223
  ],
  "overall": 0.5120549818503818,
  "responses": {
    "yield_pct": 69.34893318762082,
    "impurity_pct": 1.922917437992454,
    "cost": 9.9999999999989
  },
  "individual": {
    "yield_pct": 0.3116311062540272,
    "impurity_pct": 0.4308330248030184,
    "cost": 0.9999999999994502
  },
  "warnings": []
}
```

## Plot data

Headless plotting cores as JSON, for a frontend to render (Plotly/Vega) — no matplotlib in the service. Pareto / main-effects / half-normal plots need no endpoint; they render directly from `/v1/analysis/fit`'s `terms` array.

### Response-surface mesh

A `resolution × resolution` grid of the fitted surface over two axes (`fixed` pins any others), as natural-unit `x`/`y`/`z` meshes.

```bash
curl -s -X POST http://localhost:8000/v1/plot-data/surface \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic",
  "x": "temp",
  "y": "time",
  "resolution": 5
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "x": [
    [
      20.0,
      35.0,
      "… 3 more"
    ],
    [
      20.0,
      35.0,
      "… 3 more"
    ],
    "… 3 more"
  ],
  "y": [
    [
      0.0,
      0.0,
      "… 3 more"
    ],
    [
      2.5,
      2.5,
      "… 3 more"
    ],
    "… 3 more"
  ],
  "z": [
    [
      62.44117719429315,
      65.33469723223472,
      "… 3 more"
    ],
    [
      64.13914284446697,
      67.30762144660903,
      "… 3 more"
    ],
    "… 3 more"
  ]
}
```

### Interaction lines

One line per trace level of the `trace` factor across the `x` factor — the data behind an interaction plot.

```bash
curl -s -X POST http://localhost:8000/v1/plot-data/interactions \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "response": "yield",
  "model": "quadratic",
  "x": "temp",
  "trace": "time",
  "resolution": 5
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "x": [
    20.0,
    35.0,
    50.0,
    "… 2 more"
  ],
  "lines": [
    {
      "trace_value": 0.0,
      "z": [
        62.44117719429315,
        65.33469723223472,
        66.6437213722321,
        "… 2 more"
      ]
    },
    {
      "trace_value": 10.0,
      "z": [
        63.03846020554647,
        67.03181450029003,
        69.44067289708941,
        "… 2 more"
      ]
    }
  ]
}
```

### Ternary (mixture) surface

A 3-component Scheffé blending surface sampled over the simplex, as flat `x`/`y`/`z` plus the barycentric `points` — feed straight to a ternary contour renderer. Requires a 3-component mixture fit.

```bash
curl -s -X POST http://localhost:8000/v1/plot-data/ternary \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "extreme_vertices_3",
    "factors": [
      {
        "type": "mixture",
        "name": "polymer",
        "low": 0.2,
        "high": 0.6,
        "units": null
      },
      {
        "type": "mixture",
        "name": "solvent",
        "low": 0.2,
        "high": 0.6,
        "units": null
      },
      {
        "type": "mixture",
        "name": "additive",
        "low": 0.0,
        "high": 0.4,
        "units": null
      }
    ],
    "runs": [
      {
        "polymer": 0.2,
        "solvent": 0.4,
        "additive": 0.4,
        "gel_strength": 48.0
      },
      {
        "polymer": 0.2,
        "solvent": 0.6,
        "additive": 0.2,
        "gel_strength": 51.3
      },
      {
        "polymer": 0.4,
        "solvent": 0.2,
        "additive": 0.4,
        "gel_strength": 50.6
      },
      {
        "polymer": 0.4,
        "solvent": 0.6,
        "additive": 0.0,
        "gel_strength": 60.9
      },
      {
        "polymer": 0.6,
        "solvent": 0.2,
        "additive": 0.2,
        "gel_strength": 58.2
      },
      {
        "polymer": 0.6,
        "solvent": 0.4,
        "additive": 0.0,
        "gel_strength": 65.5
      },
      {
        "polymer": 0.4000000000000001,
        "solvent": 0.39999999999999997,
        "additive": 0.19999999999999998,
        "gel_strength": 57.8
      }
    ],
    "point_types": [
      "vertex",
      "vertex",
      "vertex",
      "vertex",
      "vertex",
      "vertex",
      "centroid"
    ],
    "meta": {
      "generator": "extreme_vertices",
      "include_centroid": true,
      "n_vertices": 6
    }
  },
  "response": "gel_strength",
  "model": "scheffe-quadratic",
  "resolution": 6
}'
```

`200`

```json
{
  "x": [
    0.5,
    0.5833333333333334,
    0.6666666666666666,
    0.75,
    "… 24 more"
  ],
  "y": [
    0.8660254037844386,
    0.7216878364870322,
    0.5773502691896257,
    0.4330127018922193,
    "… 24 more"
  ],
  "z": [
    17.400000000000063,
    27.226388888888934,
    34.20555555555559,
    38.337500000000034,
    "… 24 more"
  ],
  "points": [
    [
      0.0,
      0.0,
      1.0
    ],
    [
      0.0,
      0.16666666666666666,
      0.8333333333333334
    ],
    [
      0.0,
      0.3333333333333333,
      0.6666666666666666
    ],
    [
      0.0,
      0.5,
      0.5
    ],
    "… 24 more"
  ]
}
```

### Alias matrix

The alias/correlation structure of a model on a design (no `response`): `absolute` reports magnitudes. Reveals which effects are confounded.

```bash
curl -s -X POST http://localhost:8000/v1/plot-data/alias \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "name": "central_composite_k2",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80,
        "units": "C"
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10,
        "units": null
      }
    ],
    "runs": [
      {
        "std_order": 0,
        "temp": 20.0,
        "time": 0.0,
        "yield": 62.26206325745263
      },
      {
        "std_order": 7,
        "temp": 50.0,
        "time": 10.0,
        "yield": 69.57217138506148
      },
      {
        "std_order": 1,
        "temp": 20.0,
        "time": 10.0,
        "yield": 62.93110209512028
      },
      "… 10 more"
    ],
    "point_types": [
      "factorial",
      "axial",
      "factorial",
      "… 10 more"
    ],
    "meta": {
      "generator": {
        "name": "central_composite",
        "parameters": {
          "alpha": "faced",
          "center": 5,
          "fraction": null
        }
      },
      "alpha": 1.0,
      "axial_extrapolates": false,
      "randomized": true,
      "random_seed": 20260710
    }
  },
  "model": {
    "order": 2,
    "interactions": true
  },
  "absolute": true
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`200`

```json
{
  "labels": [
    "temp",
    "time",
    "temp:time",
    "temp^2",
    "time^2"
  ],
  "matrix": [
    [
      0.9999999999999999,
      0.0,
      0.0,
      0.0,
      2.5216315845372088e-17
    ],
    [
      0.0,
      0.9999999999999999,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      1.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0,
      0.3809523809523809
    ],
    [
      2.5216315845372088e-17,
      0.0,
      0.0,
      0.3809523809523809,
      1.0
    ]
  ]
}
```

## Errors

Every non-2xx response is one envelope: `{"error": {"code", "message", "errors"}}`. See [WEBSERVICE_API.md](WEBSERVICE_API.md#errors) for the full status/code table.

### Validation error (422)

A design document that fails validation — every problem is collected in one pass into `errors` (here a run missing a factor value and a `point_types` length mismatch). The verbatim golden fixture from the contract tests.

```bash
curl -s -X POST http://localhost:8000/v1/designs/randomize \
  -H 'Content-Type: application/json' \
  -d '{
  "design": {
    "schema_version": "1.0",
    "factors": [
      {
        "type": "continuous",
        "name": "temp",
        "low": 20,
        "high": 80
      },
      {
        "type": "continuous",
        "name": "time",
        "low": 0,
        "high": 10
      }
    ],
    "runs": [
      {
        "temp": 20.0,
        "time": 1.0
      },
      {
        "temp": 21.0,
        "time": 2.0
      },
      {
        "temp": 22.0,
        "time": 3.0
      },
      "… 7 more"
    ],
    "point_types": [
      "factorial",
      "factorial",
      "factorial",
      "… 6 more"
    ]
  }
}'
```

> `design.runs` is abbreviated above for readability — POST the full design document (any generation response, or the one shown under [Attach responses](#attach-responses)).

`422`

```json
{
  "error": {
    "code": "validation_error",
    "message": "design document is invalid",
    "errors": [
      "run[3] missing value for factor 'time'",
      "'point_types' has 9 entries but there are 10 runs"
    ]
  }
}
```

### Infeasible request (422)

A domain error passed through verbatim from the library — Sobol requires power-of-two run counts, so it names the nearest valid sizes.

```bash
curl -s -X POST http://localhost:8000/v1/designs/space-filling \
  -H 'Content-Type: application/json' \
  -d '{
  "factors": [
    {
      "type": "continuous",
      "name": "temp",
      "low": 20,
      "high": 80,
      "units": "C"
    },
    {
      "type": "continuous",
      "name": "time",
      "low": 0,
      "high": 10,
      "units": "min"
    }
  ],
  "sampler": "sobol",
  "n_runs": 10
}'
```

`422`

```json
{
  "error": {
    "code": "infeasible",
    "message": "sobol requires a power-of-two n_runs; got 10 (nearest valid sizes: 8, 16)",
    "errors": []
  }
}
```
