# Saving and restoring an experiment

An experiment rarely lives inside one program. You design it here, but the run sheet has to
drive a liquid-dispensing robot; the readouts come back from a plate reader or a LIMS; and
weeks later something has to fit and report on the result. Each of those is a separate tool,
and JSON is the neutral hand-off between them — a liquid-handler's protocol generator can
read the factor ranges and per-run setpoints straight out of the same document this library
writes. This walkthrough follows a single central-composite study across those boundaries:
turning a `Design` into JSON so another tool can consume it, checking a document is sound
before you act on it, and loading one back so the fit picks up exactly where it left off.

1. start from a finished experiment (design + measured response),
2. serialize it to a dict and to a JSON string,
3. validate the document before trusting it,
4. deserialize it and confirm the round-trip is exact,
5. and round-trip a bare factor set as a reusable template.

The rule to keep in mind throughout: **the dict is the source of truth.** `to_dict()` and
`from_dict()` do the DoE-aware work; wrapping `json.dumps` / `json.loads` around the dict is
left to you, so the document can travel however the receiving tool expects it — a file
dropped in a watch folder, a database column, an HTTP request body to an instrument's API.

> Every console output below is real: it is produced by running the snippets via
> `scripts/build_workflow4_assets.py`. The response is the same synthetic quadratic surface
> used in [WORKFLOW.md](WORKFLOW.md) so the numbers line up; replace it with your own
> measurements and the same calls apply.

## 1. Start from a finished experiment

We pick up where [WORKFLOW.md](WORKFLOW.md) leaves off: a three-factor faced central
composite, randomized, with a measured yield attached to every run. This is the object worth
persisting — not just the plan, but the plan *and* its results together, so nothing has to be
re-paired by hand later.

```python
import json
import numpy as np
from doe import ContinuousFactor, central_composite

factors = [
    ContinuousFactor("temperature", low=45, high=75, units="C"),
    ContinuousFactor("time", low=20, high=60, units="min"),
    ContinuousFactor("catalyst", low=0.5, high=2.5, units="g/L"),
]
design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260707)

# ... run the experiment; here a synthetic surface stands in for the lab ...
coded = design.coded()
rng = np.random.default_rng(42)
yield_pct = (
    78 + 7.5 * coded["temperature"] + 5.0 * coded["time"] + 3.0 * coded["catalyst"]
    - 8.0 * coded["temperature"] ** 2 - 5.5 * coded["time"] ** 2
    - 4.0 * coded["catalyst"] ** 2 + 2.5 * coded["temperature"] * coded["time"]
    - 1.5 * coded["time"] * coded["catalyst"] + rng.normal(0, 0.8, design.n_runs)
)
measured = design.with_response("yield_pct", yield_pct)
print(measured.runs.head(4).round(2))
```

```text
   std_order  temperature  time  catalyst  yield_pct
0         12         60.0  40.0       0.5      71.24
1         16         60.0  40.0       1.5      77.17
2          1         45.0  20.0       2.5      55.60
3          8         45.0  40.0       1.5      63.25
```

Two columns beyond the factors are already doing bookkeeping you will want to survive the
trip: `std_order` remembers each run's place in the textbook layout even after randomizing,
and `yield_pct` carries the measured response. Both are ordinary DataFrame columns, so both
ride along automatically when the design is serialized.

## 2. Serialize

`to_dict()` produces a plain, JSON-ready dictionary. Nothing about it is DoE-specific to
read — it is the factors, the run table, the point-type tags, and a `meta` block — but
producing it *is* DoE-specific: numpy scalars are coerced to native Python floats and ints so
the result survives `json.dumps` without a custom encoder.

```python
doc = measured.to_dict()
print("top-level keys:", list(doc))
print("schema_version:", doc["schema_version"])
print("first factor:", doc["factors"][0])
print("first run:", doc["runs"][0])
print("point_types:", doc["point_types"][:5], "...")

blob = json.dumps(doc, indent=2)          # -> write to a file, DB, or request body
print("JSON length (chars):", len(blob))
```

```text
top-level keys: ['schema_version', 'name', 'factors', 'runs', 'point_types', 'meta']
schema_version: 1.0
first factor: {'type': 'continuous', 'name': 'temperature', 'low': 45, 'high': 75, 'units': 'C'}
first run: {'std_order': 12, 'temperature': 60.0, 'time': 40.0, 'catalyst': 0.5, 'yield_pct': 71.24377366380355}
point_types: ['axial', 'center', 'factorial', 'axial', 'center'] ...
```

The whole study is 3,760 characters of JSON. A few things are worth pointing out, because
they are what makes the round-trip exact rather than approximate:

- **`schema_version`** (`"1.0"`) is stamped into every document. `from_dict` and the
  validator check the *major* version, so a future minor revision still loads.
- **`factors` is an ordered list.** That order fixes the column order of the model matrix,
  so preserving it preserves fitted-model behaviour.
- **`runs` are stored in natural units** — degrees, minutes, g/L — exactly as you entered
  and measured them, which is precisely what a downstream instrument needs: a dispenser wants
  real setpoints, not the internal `[-1, +1]` coding. Coded values are *derived* from the
  factor set on demand, never stored, so there is nothing to fall out of sync.
- **`point_types`** tags each run (`center`, `factorial`, `axial`). This is what
  `n_center` / `center_indices` and the lack-of-fit pure-error estimate depend on, so it has
  to survive — and it does.

To hand off, write `blob` wherever the next tool looks for it
(`pathlib.Path("study.json").write_text(blob)`); there is no special file format, so a
protocol generator, scheduler, or data-capture tool can parse it with any JSON library.

## 3. Validate before you trust it

Any document that has left your process — hand-edited, written by another tool, pulled from a
database that has seen a schema change — deserves a check before you build a robot protocol
or a model on it. `validate_design_dict` is that gate. It reports **every** problem at once
(not just the first), so a bad document can be fixed in a single pass.

```python
from doe import validate_design_dict, ValidationError

validate_design_dict(doc)         # returns None on success; raises on any problem
print("OK")
```

On the document we just produced it returns quietly. To see it earn its keep, corrupt a copy
the way a careless hand edit might — drop a factor value from one run, and put a word where a
number belongs in another:

```python
broken = json.loads(blob)
del broken["runs"][3]["catalyst"]
broken["runs"][5]["temperature"] = "hot"
try:
    validate_design_dict(broken)
except ValidationError as exc:
    print(len(exc.errors), "problem(s):")
    for err in exc.errors:
        print("  -", err)
```

```text
2 problem(s):
  - run[3] missing value for factor 'catalyst'
  - run[5] factor 'temperature': 'hot' is not numeric
```

Both faults are caught in one call, each naming the run index and factor so it is easy to
locate. By default continuous values are **not** range-checked — a central composite
deliberately pushes axial points outside the `[low, high]` box, so an out-of-range value is
normal, not an error. For a hand-authored factorial plan where no extrapolation is expected,
pass `check_ranges=True` to additionally require every value to sit inside its factor bounds.

## 4. Deserialize and confirm the round-trip

`Design.from_dict` rebuilds the object from the document. The promise is not "something
close" but **exact reproduction** — same runs, same tags, same everything a fit is computed
from:

```python
from doe import Design, fit_ols

restored = Design.from_dict(json.loads(blob))
print("runs identical:", restored.runs.equals(measured.runs))
print("point_types identical:", restored.point_types == measured.point_types)
print("n_center:", restored.n_center, "center_indices:", restored.center_indices)
```

```text
runs identical: True
point_types identical: True
n_center: 5 center_indices: [ 1  4  5 17 18]
```

The run table matches to the value, the center-point bookkeeping is intact, and — the point
of the whole exercise — a model fit to the restored design is indistinguishable from one fit
before saving:

```python
fit_before = fit_ols(measured, "yield_pct", model="quadratic")
fit_after = fit_ols(restored, "yield_pct", model="quadratic")
print("R2 before:", round(fit_before.r_squared, 6))
print("R2 after :", round(fit_after.r_squared, 6))
print("coefficients identical:",
      np.allclose(fit_before.summary()["coefficient"],
                  fit_after.summary()["coefficient"]))
```

```text
R2 before: 0.997535
R2 after : 0.997535
coefficients identical: True
```

Because coded values and the model matrix are *derived* from the stored factor set rather
than saved alongside it, there is no way for them to drift: restore the factors and the runs,
and every downstream number follows. The save/load boundary is invisible to the analysis.

## 5. Round-trip a factor set as a template

You do not have to serialize a whole experiment. A single factor — or a `FactorSet` — round-
trips the same way, which is the natural way to keep a *reusable template* of "the knobs and
ranges we always run" and stamp out fresh designs from it. `factor_from_dict` dispatches on
the `"type"` discriminator, so it reconstructs continuous, categorical, or mixture factors
without you having to know which you have:

```python
from doe import factor_from_dict

print(factors[0].to_dict())
one = factor_from_dict(factors[0].to_dict())
print("round-trips equal:", one == factors[0])
```

```text
{'type': 'continuous', 'name': 'temperature', 'low': 45, 'high': 75, 'units': 'C'}
round-trips equal: True
```

That is the whole story: `to_dict()` on the way out, `validate_design_dict` at any trust
boundary, `from_dict` / `factor_from_dict` on the way back, and `json` for transport in
between. For the exact document shape, the field-by-field guarantees, and how this is
expected to grow toward lab-automation use, see the [serialization reference](SERIALIZATION.md).
```