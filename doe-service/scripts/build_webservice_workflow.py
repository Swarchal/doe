"""Run the docs/WEBSERVICE_WORKFLOW.md walkthrough through the real HTTP service and
capture its console outputs.

This is the provenance for docs/WEBSERVICE_WORKFLOW.md — the service mirror of
docs/WORKFLOW.md. It reproduces the same reaction-optimization study (same factors, same
faced central-composite design, same seeded synthetic response), but every step goes
through the FastAPI app via ``TestClient`` (no network) instead of the in-process
library, so the printed request/response blocks are exactly what a client would send and
receive. Because the response surface and randomization are seeded, the numbers are
identical to WORKFLOW.md and reproducible run to run.

Run from the ``doe-service`` directory::

    uv run --extra dev python scripts/build_webservice_workflow.py

It prints one labelled block per walkthrough section for transcription into the doc
(text-only, like WORKFLOW4.md — the service is headless, so plots are described via the
plot-data endpoints rather than rendered here).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from doe_service.main import create_app

client = TestClient(create_app())


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(path, json=body)
    if resp.status_code != 200:
        raise SystemExit(f"{path} -> {resp.status_code}: {resp.text}")
    return resp.json()


def coded_columns(design: dict[str, Any]) -> dict[str, np.ndarray]:
    """Recover coded [-1, 1] columns from the natural-unit runs the service returns."""
    cols: dict[str, np.ndarray] = {}
    for f in design["factors"]:
        center = (f["low"] + f["high"]) / 2
        half = (f["high"] - f["low"]) / 2
        cols[f["name"]] = np.array([run[f["name"]] for run in design["runs"]]) - center
        cols[f["name"]] = cols[f["name"]] / half
    return cols


# --------------------------------------------------------------------------- #
# 1. Factors
# --------------------------------------------------------------------------- #
banner("Section 1: factors")
factors = [
    {"type": "continuous", "name": "temperature", "low": 45, "high": 75, "units": "C"},
    {"type": "continuous", "name": "time", "low": 20, "high": 60, "units": "min"},
    {"type": "continuous", "name": "catalyst", "low": 0.5, "high": 2.5, "units": "g/L"},
]
print(json.dumps(factors, indent=2))

# --------------------------------------------------------------------------- #
# 2. Generate + randomize
# --------------------------------------------------------------------------- #
banner("Section 2: central-composite, then randomize")
gen = post(
    "/v1/designs/central-composite",
    {"factors": factors, "alpha": "faced", "center": 5},
)
rand = post("/v1/designs/randomize", {"design": gen["design"], "seed": 20260707})
design = rand["design"]
print("n_runs:", len(design["runs"]))
print("n_center:", sum(1 for pt in design["point_types"] if pt == "center"))
print("meta.random_seed:", design["meta"].get("random_seed"))
print("first 8 runs (natural units):")
for run in design["runs"][:8]:
    print(
        f"  std_order={run['std_order']:>2}  "
        f"temperature={run['temperature']:.1f}  "
        f"time={run['time']:.1f}  catalyst={run['catalyst']:.2f}"
    )

# --------------------------------------------------------------------------- #
# 3. Attach the (synthetic) measured response
# --------------------------------------------------------------------------- #
banner("Section 3: attach the response")
coded = coded_columns(design)
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
    + rng.normal(0, 0.8, len(design["runs"]))
)
# Rounded to 3 decimals — exactly the values transcribed into the doc, so every
# downstream number reproduces from what a reader pastes.
yields = [round(float(v), 3) for v in yield_pct]
attached = post(
    "/v1/designs/responses",
    {"design": design, "responses": {"yield_pct": yields}},
)
design = attached["design"]
print("yield_pct (run order, as transcribed into the doc):")
print(" ", yields)

# --------------------------------------------------------------------------- #
# 4. Fit a quadratic model
# --------------------------------------------------------------------------- #
banner("Section 4: fit quadratic")
fit = post(
    "/v1/analysis/fit",
    {"design": design, "response": "yield_pct", "model": "quadratic"},
)
print(f"r_squared     = {fit['r_squared']:.3f}")
print(f"adjusted_r2   = {fit['adjusted_r2']:.3f}")
print(f"{'term':<22}{'coefficient':>12}{'effect':>10}{'std_error':>11}{'t':>9}{'p':>9}")
for t in fit["terms"]:
    print(
        f"{t['term']:<22}{t['coefficient']:>12.2f}{t['effect']:>10.2f}"
        f"{t['std_error']:>11.2f}{t['t']:>9.2f}{t['p']:>9.2f}"
    )

# --------------------------------------------------------------------------- #
# 5. Trust checks: ANOVA + lack-of-fit + VIF
# --------------------------------------------------------------------------- #
banner("Section 5: anova + lack-of-fit + vif")
anova = post(
    "/v1/analysis/anova",
    {"design": design, "response": "yield_pct", "model": "quadratic"},
)
print(f"predicted_r2 = {anova['predicted_r2']:.3f}   press = {anova['press']:.3f}")
print(f"{'term':<22}{'ss':>10}{'df':>5}{'ms':>10}{'F':>11}{'p':>8}")
for r in anova["rows"]:
    ms = "" if r["ms"] is None else f"{r['ms']:.3f}"
    f_ = "" if r["f"] is None else f"{r['f']:.3f}"
    p_ = "" if r["p"] is None else f"{r['p']:.3f}"
    print(f"{r['term']:<22}{r['ss']:>10.3f}{r['df']:>5.0f}{ms:>10}{f_:>11}{p_:>8}")
lof = anova["lack_of_fit"]
print("lack_of_fit:", json.dumps({k: round(v, 3) for k, v in lof.items()}))

diag = post(
    "/v1/analysis/diagnostics",
    {"design": design, "model": {"order": 2, "interactions": True}},
)
print("VIF:")
for term, v in diag["vif"].items():
    print(f"  {term:<22}{v:.2f}")

# --------------------------------------------------------------------------- #
# 6. Operating point
# --------------------------------------------------------------------------- #
banner("Section 6: stationary point + optimum")
stat = post(
    "/v1/optimize/stationary-point",
    {"design": design, "response": "yield_pct", "model": "quadratic"},
)
print("stationary:", stat["kind"], stat["natural"], "->", round(stat["response"], 2))
print("           coded:", [round(c, 3) for c in stat["coded"]],
      "eigenvalues:", [round(e, 2) for e in stat["eigenvalues"]])
opt = post(
    "/v1/optimize/optimum",
    {"design": design, "response": "yield_pct", "model": "quadratic", "maximize": True},
)
print("optimum:", opt["natural"], "->", round(opt["response"], 2),
      "at_bound:", opt["at_bound"])

# --------------------------------------------------------------------------- #
# 7. Confirmation prediction
# --------------------------------------------------------------------------- #
banner("Section 7: confirmation prediction")
pred = post(
    "/v1/analysis/predict",
    {
        "design": design,
        "response": "yield_pct",
        "model": "quadratic",
        "points": [opt["natural"]],
    },
)
print("predicted yield at optimum:", round(pred["predictions"][0], 2))
print("(v1 /predict returns the point estimate; the library's prediction interval is")
print(" not yet surfaced on the wire — see the doc's closing note.)")
