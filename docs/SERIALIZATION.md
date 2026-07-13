# JSON serialization

How DoE objects are saved to and loaded from JSON, and how that is expected to grow
toward lab-automation use (protocol generation, readout ingestion, optimization
recommendations).

JSON is the default interchange format because the same artifact may move between a
Python backend, a web UI, automation software, data-capture tools, and later
analysis/reporting pipelines.

## What's serialized today

`FactorSet` and `Design` round-trip through plain dicts that are directly JSON
serializable. The **dict is the source of truth**; calling `json.dumps` /
`json.loads` around it is left to the caller.

```python
from doe import Design, FactorSet, factor_from_dict
import json

blob = json.dumps(design.to_dict())
restored = Design.from_dict(json.loads(blob))
```

### API

| Method | Notes |
| --- | --- |
| `ContinuousFactor.to_dict()` / `.from_dict(d)` | |
| `CategoricalFactor.to_dict()` / `.from_dict(d)` | |
| `factor_from_dict(d)` | Dispatches on the `"type"` discriminator. Exported from `doe`. |
| `FactorSet.to_dict()` / `.from_dict(d)` | |
| `Design.to_dict()` / `.from_dict(d)` | |
| `validate_design_dict(d, *, check_ranges=False)` | Validates a design dict; raises `ValidationError`. Exported from `doe`. |

`doe.design.SCHEMA_VERSION` (currently `"1.0"`) is stamped into every design dict.

### Validation

`validate_design_dict` checks a design document is structurally sound and internally
consistent before it is executed or analysed — a supported `schema_version`, a non-empty
uniquely-named factor list, every run carrying a value for every factor, categorical
values drawn from the declared levels, and `point_types` aligned to the runs. It collects
**all** problems and raises a single `ValidationError` whose `.errors` lists them, so a
bad document can be fixed in one pass.

Continuous values are **not** range-checked by default: response-surface designs
(central composite) deliberately place axial points outside the `[low, high]` box, so an
out-of-range value is normal there. Pass `check_ranges=True` to additionally require every
continuous value to fall within its factor bounds — useful for hand-authored factorial
plans where no extrapolation is expected. Validation has no external JSON-Schema
dependency; it is plain Python, keeping the core on the scipy stack.

### Round-trip guarantee

`Design.from_dict(d.to_dict())` reproduces a design exactly. Specifically it preserves:

- the **ordered** factor set (order fixes model-matrix column order);
- the full `runs` table in **natural** units, including any `std_order` column added
  by `randomize()` and any response columns appended after experiments;
- `point_types`, so `n_center` / `center_indices` and the lack-of-fit pure-error
  estimate are unchanged;
- `meta`, which carries the generating request (the `generator` block) and — after
  `randomize()` — the `random_seed`, so the run order is both preserved (`std_order`)
  and reproducible (the seed);
- and therefore **fitted-model behaviour**, since coded values and the model matrix
  are *derived* from the factor set, not stored.

Numpy scalars are coerced to native Python types on the way out (without precision
loss), so the result is JSON-safe. Covered by `tests/test_serialization.py`.

### Design dict shape

Actual output of `Design.to_dict()` for a randomized central-composite design
(run table trimmed to two rows):

```json
{
  "schema_version": "1.0",
  "name": "central_composite_k2",
  "factors": [
    { "type": "continuous", "name": "temp", "low": 20.0, "high": 80.0, "units": "C" },
    { "type": "continuous", "name": "time", "low": 0.0, "high": 10.0, "units": null }
  ],
  "runs": [
    { "std_order": 7, "temp": 50.0, "time": 10.0 },
    { "std_order": 4, "temp": 20.0, "time": 5.0 }
  ],
  "point_types": [
    "axial", "axial", "factorial", "factorial", "factorial",
    "axial", "axial", "center", "factorial", "center"
  ],
  "meta": {
    "generator": {
      "name": "central_composite",
      "parameters": { "alpha": "faced", "center": 2, "fraction": null }
    },
    "alpha": 1.0,
    "axial_extrapolates": false,
    "randomized": true,
    "random_seed": 123
  }
}
```

Shape conventions:

- `factors` is a flat list with a `"type"` discriminator; categorical factors carry
  `"levels"` instead of `low`/`high`. Mixture components (`"type": "mixture"`) carry
  proportion bounds `low`/`high` in `[0, 1]` (defaulting to the full `[0, 1]`), and a
  factor set is either all-mixture or mixture-free.
- `runs` records are flat `{column: value}` maps (factor columns plus any
  `std_order` / response columns), not a nested `values` object.
- `point_types` is a single array aligned to `runs`, not a per-run field.
- `whole_plots` (split-plot designs only) is a single array of integer plot ids aligned to
  `runs`, one per run; runs sharing an id form one whole plot. It is **emitted only when set**,
  so fully-randomized designs serialize byte-for-byte as before, and it survives
  `replicate`/`randomize`/`project` like `point_types`. `validate_design_dict` checks it is a
  list of integers the same length as `runs`.
- `hard_to_change: true` appears on a `continuous`/`categorical` factor when it is a whole-plot
  (hard-to-change) factor in a split-plot design. It too is **emitted only when `true`** (so
  pre-split-plot documents are unchanged) and reads back as `False` when absent.
- `meta["generator"]` records the generating *request* as `{"name", "parameters"}` —
  the call that regenerates the design (e.g. the defining relation strings of a
  fractional factorial, or `alpha="rotatable"` for a CCD). Values *resolved* from the
  request (the numeric `alpha` above) and randomization state sit alongside as plain
  `meta` keys. Optimal designs record their search the same flat way (`criterion`,
  `model`, `seed`, ...), space-filling designs their sampler (`sampler`, `seed`, ...).
- every recorded seed is a concrete integer: `randomize()`, the optimal-design
  search, and the space-filling samplers draw one when none is given, so a
  serialized design never carries an unreproducible `null` seed.

## Why it's shaped this way

Two principles drive the choices above:

- **Natural values are the public contract.** Run values are stored in natural
  units; coded (`[-1, +1]`) values are derived from the `FactorSet` and deliberately
  not serialized, so there is a single source of truth. Likewise categorical levels
  are stored as natural labels, not encoded contrast columns — deviation/effect
  coding is regenerated at analysis time.
- **Keep the statistical design independent of lab automation.** A design run says
  "temperature is 40 C and buffer is B"; a protocol step says "transfer 12.5 µL from
  well A3 to C7". The same design may run on different robots, decks, labware, or
  stock concentrations, so these belong in separate layers:

| Layer | Purpose | Example contents | Status |
| --- | --- | --- | --- |
| Design spec | Reconstruct the intended experiment | Factor definitions, the `meta["generator"]` block (name + requested parameters), seeds | **implemented** |
| Generated design | Immutable plan to execute | Run table, natural values, point types, run order | **implemented** |
| Execution protocol | Carry out the design | Source/destination wells, volumes, stock concentrations, deck layout | future |
| Observed results | Attach readouts to runs | Response values, units, timestamps, QC flags, failed runs | future |
| Analysis result | Reproduce conclusions | Coefficients, ANOVA, optimum, desirability settings | derived (see below) |

## Analysis results

Analysis results (`FitResult`, `Optimum`, etc.) are *derived* from a design plus a
response and need not be persisted as a source of truth — storing the inputs and
re-fitting is enough. To make re-fitting possible, `FitResult` records the resolved
model spec (`order` and `interactions`), so a serialized fit can be rebuilt with the
same terms (the convenience name `"linear"` / `"quadratic"` is not itself stored).
Serialize the full result arrays only when an audit snapshot must survive future
algorithm changes.

Optimization **goals** are data, not derived results. `ResponseGoal.to_dict()` serializes
the goal definition (`goal`, `low`, `high`, `target`, `weight`) so a recommendation is
reproducible; the bound `FitResult` is omitted, since it is re-fit from the design plus a
response. `ResponseGoal.from_dict(d, result)` takes that re-fitted result back in.

## Roadmap

Planned work, roughly in order. `schema_version` exists so these can land without
breaking older documents.

Already landed (documented above): **document validation** (`validate_design_dict`),
**optimization goals as data** (`ResponseGoal.to_dict`/`from_dict`), and the
**generator spec** — every generator records its regenerating request in
`meta["generator"]` (or, for optimal/space-filling designs, flat search/sampler keys)
with a concretely drawn seed, so the intended experiment survives serialization.

1. **Richer per-run fields.** Stable `run_id`s (so readouts, protocol logs, and
   failures join safely instead of relying on row position / `std_order`), per-run
   `replicate` / `run_order`, and promotion of the `generator` block from `meta` to a
   first-class document field. Target shape:

   ```json
   {
     "schema_version": "1.0",
     "design": {
       "name": "ccd_example",
       "generator": {
         "name": "central_composite",
         "parameters": { "alpha": "rotatable", "center": 5 }
       },
       "randomized": true,
       "random_seed": 123,
       "factors": [
         { "name": "temperature", "type": "continuous", "low": 20, "high": 80, "units": "C" },
         { "name": "buffer", "type": "categorical", "levels": ["A", "B", "C"] }
       ],
       "runs": [
         {
           "run_id": "run_001",
           "std_order": 1,
           "run_order": 7,
           "point_type": "factorial",
           "replicate": 1,
           "values": { "temperature": 20, "buffer": "A" }
         }
       ]
     }
   }
   ```

2. **Results / readout attachment.** Observed readouts stored separately and joined
   by `run_id`, carrying response units, status, timestamps, and QC metadata:

   ```json
   {
     "schema_version": "1.0",
     "design_hash": "sha256:...",
     "results": [
       {
         "run_id": "run_001",
         "status": "complete",
         "responses": {
           "yield":  { "value": 72.4, "units": "%" },
           "purity": { "value": 0.91, "units": "fraction" }
         }
       }
     ]
   }
   ```

3. **Richer factor / automation metadata.** Precision, stock concentration, dilution
   constraints, min/max pipettable volumes, factor role, allowed ranges.
4. **`ExperimentPlan`.** A higher-level object combining a design with execution
   metadata, observations, and analysis results — introduced only once execution
   metadata is concrete enough to model without overfitting to one robot or labware
   setup, and recording its source design (ideally via a content hash) for
   traceability.
