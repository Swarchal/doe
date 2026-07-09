"""Run the docs/WORKFLOW4.md walkthrough and capture its real console outputs.

This is the provenance for the outputs embedded in docs/WORKFLOW4.md: it reproduces the
serialize/deserialize walkthrough verbatim (same factors, same seed), printing every
console block for transcription. The walkthrough is text-only (no figures). Run with:
uv run python scripts/build_workflow4_assets.py
"""

from __future__ import annotations

import json

import numpy as np

from doe import (
    ContinuousFactor,
    Design,
    ValidationError,
    central_composite,
    factor_from_dict,
    fit_ols,
    validate_design_dict,
)


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1. A finished experiment: design + measured response
# --------------------------------------------------------------------------- #
banner("Section 1: a finished experiment")

factors = [
    ContinuousFactor("temperature", low=45, high=75, units="C"),
    ContinuousFactor("time", low=20, high=60, units="min"),
    ContinuousFactor("catalyst", low=0.5, high=2.5, units="g/L"),
]
design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260707)

coded = design.coded()
rng = np.random.default_rng(42)
yield_pct = (
    78
    + 7.5 * coded["temperature"]
    + 5.0 * coded["time"]
    + 3.0 * coded["catalyst"]
    - 8.0 * coded["temperature"] ** 2
    - 5.5 * coded["time"] ** 2
    - 4.0 * coded["catalyst"] ** 2
    + 2.5 * coded["temperature"] * coded["time"]
    - 1.5 * coded["time"] * coded["catalyst"]
    + rng.normal(0, 0.8, design.n_runs)
)
measured = design.with_response("yield_pct", yield_pct)
print(measured.runs.head(4).round(2))

# --------------------------------------------------------------------------- #
# 2. Serialize to a dict and to a JSON string / file
# --------------------------------------------------------------------------- #
banner("Section 2: serialize")

doc = measured.to_dict()
print("top-level keys:", list(doc))
print("schema_version:", doc["schema_version"])
print("first factor:", doc["factors"][0])
print("first run:", doc["runs"][0])
print("point_types:", doc["point_types"][:5], "...")

blob = json.dumps(doc, indent=2)
print("\nJSON length (chars):", len(blob))
print("first 3 lines:")
for line in blob.splitlines()[:3]:
    print(line)

# --------------------------------------------------------------------------- #
# 3. Validate before trusting the document
# --------------------------------------------------------------------------- #
banner("Section 3: validate")

validate_design_dict(doc)
print("validate_design_dict(doc): OK (returned None, no exception)")

broken = json.loads(blob)
del broken["runs"][3]["catalyst"]
broken["runs"][5]["temperature"] = "hot"
try:
    validate_design_dict(broken)
except ValidationError as exc:
    print("\ncaught ValidationError with", len(exc.errors), "problem(s):")
    for err in exc.errors:
        print("  -", err)

# --------------------------------------------------------------------------- #
# 4. Deserialize and confirm the round-trip
# --------------------------------------------------------------------------- #
banner("Section 4: deserialize + round-trip")

restored = Design.from_dict(json.loads(blob))
print("runs identical:", restored.runs.equals(measured.runs))
print("point_types identical:", restored.point_types == measured.point_types)
print("n_center:", restored.n_center, "center_indices:", restored.center_indices)

fit_before = fit_ols(measured, "yield_pct", model="quadratic")
fit_after = fit_ols(restored, "yield_pct", model="quadratic")
print("R2 before:", round(fit_before.r_squared, 6))
print("R2 after :", round(fit_after.r_squared, 6))
print(
    "coefficients identical:",
    np.allclose(fit_before.summary()["coefficient"], fit_after.summary()["coefficient"]),
)

# --------------------------------------------------------------------------- #
# 5. Just the factor set (a reusable template)
# --------------------------------------------------------------------------- #
banner("Section 5: a single factor round-trips too")

one = factor_from_dict(factors[0].to_dict())
print(factors[0].to_dict())
print("round-trips equal:", one == factors[0])
