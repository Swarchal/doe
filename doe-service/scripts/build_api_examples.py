"""Generate docs/WEBSERVICE_EXAMPLES.md — a worked, curl-runnable example per endpoint.

This is the provenance for the DoE web-service cookbook, the same way
``scripts/build_workflow*_assets.py`` are the provenance for the workflow docs: every
request/response pair in the generated markdown is *real*, produced by POSTing through
the actual FastAPI app (``TestClient``, no network) and capturing exactly what the
service returns today. Randomised endpoints are seeded, so re-runs are stable.

Run from the ``doe-service`` directory::

    uv run --extra dev python scripts/build_api_examples.py

It rewrites ``../docs/WEBSERVICE_EXAMPLES.md`` in place. When you change a response
shape, re-run it and commit the regenerated doc so the cookbook stays truthful.

Long arrays (runs, fitted values, residuals, meshes) are abbreviated to a few entries
followed by a ``"… N more"`` marker so the JSON stays readable; the abbreviation is
applied uniformly by :func:`abbreviate` and flagged in the doc's preamble.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from doe_service.main import create_app

BASE = "http://localhost:8000"
KEEP = 4  # first N entries kept when abbreviating a long list
OUT = Path(__file__).resolve().parents[2] / "docs" / "WEBSERVICE_EXAMPLES.md"

client = TestClient(create_app())
blocks: list[str] = []


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def abbreviate(value: Any, keep: int = KEEP) -> Any:
    """Recursively shorten long lists to ``keep`` entries + a ``"… N more"`` marker,
    so a captured response prints readably while staying valid JSON."""
    if isinstance(value, dict):
        return {k: abbreviate(v, keep) for k, v in value.items()}
    if isinstance(value, list):
        shortened = [abbreviate(v, keep) for v in value[:keep]]
        if len(value) > keep:
            shortened.append(f"… {len(value) - keep} more")
        return shortened
    return value


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def curl(path: str, body: dict[str, Any] | None, method: str = "POST") -> str:
    lines = [f"curl -s -X {method} {BASE}{path} \\", "  -H 'Content-Type: application/json'"]
    if body is not None:
        payload = dumps(body).replace("'", "'\\''")
        lines[-1] += " \\"
        lines.append(f"  -d '{payload}'")
    return "\n".join(lines)


def section(title: str, intro: str = "") -> None:
    blocks.append(f"## {title}\n")
    if intro:
        blocks.append(intro.strip() + "\n")


# A request carrying a design with more runs than this has its `design` abbreviated in
# the *displayed* curl (the POSTed body is always full), so the shared analysis design
# isn't re-dumped under every endpoint. Small designs (operations examples) stay whole.
_ABBREV_RUNS_OVER = 8


def _display_body(body: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    """The request as *displayed*: if it embeds a large design, shorten its
    ``runs``/``point_types``/``whole_plots`` for readability. Returns the (possibly
    trimmed) copy and whether it was trimmed."""
    if not isinstance(body, dict):
        return body, False
    design = body.get("design")
    if not (isinstance(design, dict) and len(design.get("runs", [])) > _ABBREV_RUNS_OVER):
        return body, False
    shown = {**body, "design": {**design}}
    for field in ("runs", "point_types", "whole_plots"):
        if isinstance(shown["design"].get(field), list):
            shown["design"][field] = abbreviate(shown["design"][field], 3)
    return shown, True


def example(
    title: str,
    desc: str,
    path: str,
    body: dict[str, Any] | None,
    *,
    method: str = "POST",
    keep: int = KEEP,
) -> dict[str, Any]:
    """POST ``body`` to ``path``, capture the live response, and append a doc block.

    Returns the raw (un-abbreviated) response JSON so later examples can chain off it.
    """
    if method == "POST":
        resp = client.post(path, json=body)
    else:
        resp = client.request(method, path)
    payload = resp.json()

    shown, trimmed = _display_body(body)
    blocks.append(f"### {title}\n")
    if desc:
        blocks.append(desc.strip() + "\n")
    blocks.append("```bash\n" + curl(path, shown, method) + "\n```\n")
    if trimmed:
        blocks.append(
            "> `design.runs` is abbreviated above for readability — POST the full design "
            "document (any generation response, or the one shown under "
            "[Attach responses](#attach-responses)).\n"
        )
    blocks.append(f"`{resp.status_code}`\n")
    blocks.append("```json\n" + dumps(abbreviate(payload, keep)) + "\n```\n")
    return payload


# --------------------------------------------------------------------------- #
# Reusable factor sets and designs
# --------------------------------------------------------------------------- #
def cont(name: str, low: float, high: float, **extra: Any) -> dict[str, Any]:
    return {"type": "continuous", "name": name, "low": low, "high": high, **extra}


TEMP_TIME = [cont("temp", 20, 80, units="C"), cont("time", 0, 10, units="min")]
THREE_CONT = [cont("temp", 20, 80), cont("time", 0, 10), cont("conc", 1, 5)]
MIXTURE = [
    {"type": "mixture", "name": "polymer", "low": 0.2, "high": 0.6},
    {"type": "mixture", "name": "solvent", "low": 0.2, "high": 0.6},
    {"type": "mixture", "name": "additive", "low": 0.0, "high": 0.4},
]
# simplex_lattice / simplex_centroid require the full (unconstrained) simplex.
MIXTURE_FULL = [{"type": "mixture", "name": n} for n in ("polymer", "solvent", "additive")]

PAIRS = Path(__file__).resolve().parent.parent / "tests" / "contract" / "pairs"


def pair_request(name: str) -> dict[str, Any]:
    return json.loads((PAIRS / f"{name}.json").read_text())["request"]


# The golden quadratic-yield CCD used across the analysis/optimize/plot examples,
# reused verbatim from the contract fixtures so the numbers match WEBSERVICE_API.md.
FIT_DESIGN = pair_request("fit")["design"]
DESIRABILITY_REQ = pair_request("desirability")


def build_mixed_design() -> dict[str, Any]:
    """A D-optimal design over two continuous + one categorical factor with a synthetic
    quadratic response, for the mixed categorical-optimum example."""
    factors = [
        {"type": "continuous", "name": "temp", "low": 20, "high": 80, "units": "C"},
        {"type": "continuous", "name": "time", "low": 2, "high": 10, "units": "min"},
        {"type": "categorical", "name": "catalyst", "levels": ["A", "B"]},
    ]
    gen = client.post(
        "/v1/designs/optimal",
        json={"factors": factors, "n_runs": 16, "model": "quadratic", "seed": 0},
    ).json()["design"]
    # Concave surface peaking at temp=68.75, time=9.0, with catalyst "B" adding a flat +8.
    for run in gen["runs"]:
        t = (run["temp"] - 50) / 30
        m = (run["time"] - 6) / 4
        bump = 8.0 if run["catalyst"] == "B" else 0.0
        run["yield"] = round(70 + 5 * t + 3 * m - 4 * t**2 - 2 * m**2 + bump, 3)
    return gen


def build_scheffe_design() -> dict[str, Any]:
    """A 3-component extreme-vertices mixture design with a synthetic blend response,
    for the Scheffé fit / ternary examples."""
    gen = client.post(
        "/v1/designs/extreme-vertices", json={"factors": MIXTURE}
    ).json()["design"]
    # A deterministic quadratic-ish blending response so the ternary surface is real.
    for i, run in enumerate(gen["runs"]):
        p, s, a = run["polymer"], run["solvent"], run["additive"]
        run["gel_strength"] = round(
            60 * p + 40 * s + 30 * a + 50 * p * s + 25 * s * a + 0.3 * i, 3
        )
    return gen


# =========================================================================== #
# Preamble
# =========================================================================== #
blocks.append(
    """# doe-service — API examples

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
""".rstrip()
    + "\n"
)


# =========================================================================== #
# Health
# =========================================================================== #
section(
    "Health check",
    "A cheap liveness probe — the one `GET` compute-adjacent route.",
)
example("Service health", "", "/v1/health", None, method="GET")


# =========================================================================== #
# Design generation
# =========================================================================== #
section(
    "Design generation",
    "One endpoint per generator. Each takes `{factors: [...], ...params}` and returns "
    "`{design, warnings}`, where `design` is a full design document you can feed "
    "straight into analysis or an operation.",
)

example(
    "Full factorial",
    "Every combination of factor levels — here the 2-level, 2-factor 2² design "
    "(add `\"levels\": 3` or a per-factor list for more levels).",
    "/v1/designs/full-factorial",
    {"factors": TEMP_TIME, "levels": 2},
)

example(
    "Fractional factorial",
    "A 2^(4−1) half-fraction from the defining relation `D=ABC`: four factors in eight "
    "runs.",
    "/v1/designs/fractional-factorial",
    {
        "factors": [cont(n, -1, 1) for n in ("A", "B", "C", "D")],
        "generators": ["D=ABC"],
    },
)

example(
    "Plackett–Burman",
    "A saturated, orthogonal main-effect screening design; run count comes from the "
    "smallest constructible Hadamard order that fits the factors.",
    "/v1/designs/plackett-burman",
    {"factors": [cont(n, 0, 1) for n in ("a", "b", "c", "d", "e")]},
)

example(
    "Definitive screening design",
    "A conference-matrix DSD: `2k+1` runs, main effects orthogonal to all second-order "
    "terms. Include a categorical factor and it takes the Jones–Nachtsheim categorical "
    "DSD-augment path automatically (no new endpoint).",
    "/v1/designs/definitive-screening",
    {"factors": THREE_CONT},
)

example(
    "Central composite (RSM)",
    "A rotatable central-composite design: factorial core + axial (`alpha`) points + "
    "center replicates, ready for a quadratic fit.",
    "/v1/designs/central-composite",
    {"factors": TEMP_TIME, "alpha": "rotatable", "center": 5},
)

example(
    "Box–Behnken (RSM)",
    "A 3-level second-order design with no corner runs — useful when the extreme "
    "corner combinations are infeasible. Needs ≥3 continuous factors.",
    "/v1/designs/box-behnken",
    {"factors": THREE_CONT, "center": 3},
)

example(
    "Space-filling (Latin hypercube / Sobol / Halton)",
    "Coverage-oriented sampling for computer experiments and surrogate modelling. "
    "`sampler` picks the engine; `sobol` requires power-of-two `n_runs` (it names the "
    "nearest valid sizes otherwise). Seeded for reproducibility.",
    "/v1/designs/space-filling",
    {"factors": TEMP_TIME, "sampler": "lhs", "n_runs": 8, "seed": 7},
)

example(
    "Simplex-lattice (mixture)",
    "A `{k, m}` lattice over the mixture simplex — every blend whose proportions are "
    "multiples of `1/degree`. Needs the full simplex (unconstrained components); use "
    "`extreme-vertices` when components are bounded.",
    "/v1/designs/simplex-lattice",
    {"factors": MIXTURE_FULL, "degree": 2},
)

example(
    "Simplex-centroid (mixture)",
    "The `2^k − 1` subset centroids of the (full) simplex.",
    "/v1/designs/simplex-centroid",
    {"factors": MIXTURE_FULL},
)

example(
    "Extreme vertices (mixture)",
    "The vertices of a bound-constrained simplex (McLean–Anderson XVERT) plus the "
    "centroid — the design to reach for when component bounds carve a polytope out of "
    "the full simplex.",
    "/v1/designs/extreme-vertices",
    {"factors": MIXTURE, "include_centroid": True},
)


# =========================================================================== #
# Optimal designs & candidates
# =========================================================================== #
section(
    "Computer-generated optimal designs",
    "The coordinate-exchange engine for when no named recipe fits (odd run budgets, "
    "mixed factor types, constrained regions). These return a `search` report alongside "
    "the design.",
)

example(
    "D-optimal design",
    "Build a 12-run D-optimal design for a quadratic model by coordinate exchange. "
    "`criterion` is `\"D\"` (default) or `\"I\"`; `region` (coded candidate rows) is "
    "optional — omitted, it defaults to a `candidate_grid`.",
    "/v1/designs/optimal",
    {
        "factors": TEMP_TIME,
        "n_runs": 12,
        "model": "quadratic",
        "criterion": "D",
        "n_restarts": 10,
        "seed": 42,
    },
)

example(
    "Candidate set",
    "The discrete candidate points the optimal engine searches over — request them so a "
    "client can filter to a feasible region (WORKFLOW6) before passing them back as "
    "`region`. `kind` is `\"grid\"` for box factors, `\"mixture\"` for a simplex.",
    "/v1/designs/candidates",
    {"factors": TEMP_TIME, "levels": 3},
)

section(
    "Split-plot & blocking (Phase 5)",
    "Restricted-randomization generators. Same `{factors, ...}` → `{design, warnings}` "
    "shape; the block-carrying generators add a reserved `block` categorical factor, and "
    "`split-plot` returns a `whole_plots` array (one plot id per run) that `fit-gls` "
    "consumes.",
)

example(
    "Split-plot design",
    "Cross a whole-plot design over the hard-to-change factors with a sub-plot design "
    "run inside each plot. Flag the hard-to-change factor with `\"hard_to_change\": "
    "true`.",
    "/v1/designs/split-plot",
    {
        "factors": [
            cont("temp", 100, 200, hard_to_change=True),
            cont("conc", 1, 5),
        ],
        "seed": 1,
    },
)

example(
    "Randomized complete block",
    "Each of `n_blocks` blocks sees every treatment once. Give either an explicit "
    "`factors` list or `n_treatments`.",
    "/v1/designs/randomized-complete-block",
    {"n_treatments": 3, "n_blocks": 4, "seed": 3},
)

example(
    "Latin square",
    "A `t × t` square controlling two nuisance directions (row/column blocks) at once.",
    "/v1/designs/latin-square",
    {"treatments": 4, "seed": 5},
)

example(
    "Blocked factorial",
    "A 2³ factorial split into blocks by confounding the `ABC` contrast; the full "
    "confounded set is recorded in `meta.confounded_with_blocks`.",
    "/v1/designs/blocked-factorial",
    {
        "factors": [cont(n, -1, 1) for n in ("A", "B", "C")],
        "block_generators": ["ABC"],
        "seed": 2,
    },
)


# =========================================================================== #
# Design operations
# =========================================================================== #
section(
    "Design operations",
    "Pure transformations: a design document in, a design document (or a validation "
    "verdict) out.",
)

small = client.post("/v1/designs/full-factorial", json={"factors": TEMP_TIME}).json()[
    "design"
]

example(
    "Validate a design document",
    "The one endpoint that returns `200` for an *invalid* design too — the verdict is "
    "the payload. `check_ranges` additionally flags runs outside a factor's declared "
    "range.",
    "/v1/designs/validate",
    {"design": small, "check_ranges": False},
)

example(
    "Randomize run order",
    "Shuffle run order (respecting split-plot structure when present). The seed used is "
    "echoed in `meta` so the order is reproducible.",
    "/v1/designs/randomize",
    {"design": small, "seed": 11},
)

example(
    "Replicate runs",
    "Duplicate the whole design `n` times (`each: true` instead repeats each run in "
    "place).",
    "/v1/designs/replicate",
    {"design": small, "n": 2, "each": False},
    keep=6,
)

example(
    "Project onto a subset of factors",
    "Drop columns for the factors that didn't survive screening, keeping runs and any "
    "responses — the setup for `augment`.",
    "/v1/designs/project",
    {"design": small, "factors": ["temp"]},
)

attach_target = client.post(
    "/v1/designs/full-factorial", json={"factors": TEMP_TIME}
).json()["design"]
example(
    "Attach responses",
    "Pair measured readings to runs by position (length-checked). The returned document "
    "carries the new column, ready for analysis.",
    "/v1/designs/responses",
    {"design": attach_target, "responses": {"yield": [63.1, 71.4, 68.9, 74.2]}},
)


# =========================================================================== #
# Analysis
# =========================================================================== #
section(
    "Analysis",
    "Every analysis endpoint takes `{design, response, model}` and re-fits internally — "
    "no fit handle crosses the wire. The examples below all use the same golden "
    "quadratic-`yield` central-composite design (13 runs, `temp` 20–80 °C, `time` "
    "0–10 min) that the contract tests lock in, so the numbers match "
    "[WEBSERVICE_API.md](WEBSERVICE_API.md).",
)

example(
    "Fit an OLS model",
    "Least-squares fit with coefficients, effects, standard errors, t/p values and "
    "confidence intervals per term, plus `r_squared`/`adjusted_r2` and fitted/residual "
    "vectors. Saturated fits return `null` inference columns + a `\"saturated_model\"` "
    "warning; Scheffé fits return `effect: null`.",
    "/v1/analysis/fit",
    {"design": FIT_DESIGN, "response": "yield", "model": "quadratic", "confidence": 0.95},
    keep=3,
)

example(
    "ANOVA + lack-of-fit",
    "Sequential (Type I) sums of squares, plus a lack-of-fit split when the design has "
    "replicate center points (otherwise `lack_of_fit: null` + a `\"no_pure_error\"` "
    "warning) and PRESS-based predictive R².",
    "/v1/analysis/anova",
    {"design": FIT_DESIGN, "response": "yield", "model": "quadratic"},
    keep=8,
)

example(
    "Predict at new points",
    "Evaluate the fitted model at natural-unit points — the workhorse for grid searches "
    "over constrained regions. Each point must cover every factor.",
    "/v1/analysis/predict",
    {
        "design": FIT_DESIGN,
        "response": "yield",
        "model": "quadratic",
        "points": [{"temp": 55, "time": 4.2}, {"temp": 20, "time": 10}],
    },
)

example(
    "Design diagnostics",
    "Judge a design against a model *before* running it — no `response` needed. Returns "
    "D/A/G/I efficiencies, the condition number, per-term VIFs, the correlation (alias) "
    "matrix, and per-run leverage.",
    "/v1/analysis/diagnostics",
    {"design": FIT_DESIGN, "model": {"order": 2, "interactions": True}},
    keep=6,
)

example(
    "Coverage metrics",
    "Model-free space-filling quality: `discrepancy` (`method` one of "
    "`CD`/`WD`/`MD`/`L2-star`) and `maximin_distance`, both on the coded design rescaled "
    "to the unit cube.",
    "/v1/analysis/coverage",
    {"design": FIT_DESIGN, "method": "CD"},
)

# Split-plot fit reuses the fit-gls contract fixture (a design carrying whole_plots).
example(
    "Split-plot fit (REML/GLS)",
    "The split-plot front door: requires a design carrying `whole_plots`, estimates the "
    "variance ratio by REML, and returns everything `fit` does plus the whole-plot "
    "variance component (`sigma2_wp`) and two-stratum degrees of freedom (`dof_terms`) "
    "— the coarser standard errors OLS understates.",
    "/v1/analysis/fit-gls",
    pair_request("fit_gls"),
    keep=3,
)


# =========================================================================== #
# Optimization
# =========================================================================== #
section(
    "Optimization",
    "Read the optimum off a fitted quadratic surface. The first two reuse the golden "
    "`yield` design; `desirability` balances three competing responses on one design.",
)

example(
    "Stationary point + canonical analysis",
    "Solve `−½ B⁻¹ b` for the surface's stationary point and classify it "
    "(`maximum`/`minimum`/`saddle`) via the eigen-decomposition of the quadratic form.",
    "/v1/optimize/stationary-point",
    {"design": FIT_DESIGN, "response": "yield", "model": "quadratic"},
)

example(
    "Constrained optimum",
    "Multistart L-BFGS-B search over the coded box (or explicit `bounds`); `at_bound` "
    "flags when the optimum sits on a constraint.",
    "/v1/optimize/optimum",
    {"design": FIT_DESIGN, "response": "yield", "model": "quadratic", "maximize": True},
)

MIXED_DESIGN = build_mixed_design()
example(
    "Mixed continuous/categorical optimum",
    "`/optimum` has no coded box for a categorical factor. This optimizes the continuous "
    "factors *exactly* within each combination of categorical levels and returns the best; "
    "`levels` names the winning level(s) and `settings` merges them with the continuous "
    "values. An all-continuous fit is accepted too (empty `levels`).",
    "/v1/optimize/categorical-optimum",
    {"design": MIXED_DESIGN, "response": "yield", "model": "quadratic", "maximize": True},
)

example(
    "Desirability (multi-response)",
    "Derringer–Suich geometric-mean desirability across several goals "
    "(`max`/`min`/`target`), each re-fit on the shared design. Returns the best "
    "operating point plus per-response predictions and individual desirabilities.",
    "/v1/optimize/desirability",
    DESIRABILITY_REQ,
)


# =========================================================================== #
# Plot data
# =========================================================================== #
section(
    "Plot data",
    "Headless plotting cores as JSON, for a frontend to render (Plotly/Vega) — no "
    "matplotlib in the service. Pareto / main-effects / half-normal plots need no "
    "endpoint; they render directly from `/v1/analysis/fit`'s `terms` array.",
)

example(
    "Response-surface mesh",
    "A `resolution × resolution` grid of the fitted surface over two axes (`fixed` "
    "pins any others), as natural-unit `x`/`y`/`z` meshes.",
    "/v1/plot-data/surface",
    {
        "design": FIT_DESIGN,
        "response": "yield",
        "model": "quadratic",
        "x": "temp",
        "y": "time",
        "resolution": 5,
    },
    keep=2,
)

example(
    "Interaction lines",
    "One line per trace level of the `trace` factor across the `x` factor — the data "
    "behind an interaction plot.",
    "/v1/plot-data/interactions",
    {
        "design": FIT_DESIGN,
        "response": "yield",
        "model": "quadratic",
        "x": "temp",
        "trace": "time",
        "resolution": 5,
    },
    keep=3,
)

SCHEFFE_DESIGN = build_scheffe_design()
example(
    "Ternary (mixture) surface",
    "A 3-component Scheffé blending surface sampled over the simplex, as flat "
    "`x`/`y`/`z` plus the barycentric `points` — feed straight to a ternary contour "
    "renderer. Requires a 3-component mixture fit.",
    "/v1/plot-data/ternary",
    {
        "design": SCHEFFE_DESIGN,
        "response": "gel_strength",
        "model": "scheffe-quadratic",
        "resolution": 6,
    },
    keep=4,
)

example(
    "Alias matrix",
    "The alias/correlation structure of a model on a design (no `response`): `absolute` "
    "reports magnitudes. Reveals which effects are confounded.",
    "/v1/plot-data/alias",
    {
        "design": FIT_DESIGN,
        "model": {"order": 2, "interactions": True},
        "absolute": True,
    },
    keep=6,
)


# =========================================================================== #
# Errors
# =========================================================================== #
section(
    "Errors",
    "Every non-2xx response is one envelope: `{\"error\": {\"code\", \"message\", "
    "\"errors\"}}`. See [WEBSERVICE_API.md](WEBSERVICE_API.md#errors) for the full "
    "status/code table.",
)

_validation_pair = json.loads((PAIRS / "validation_error.json").read_text())
example(
    "Validation error (422)",
    "A design document that fails validation — every problem is collected in one pass "
    "into `errors` (here a run missing a factor value and a `point_types` length "
    "mismatch). The verbatim golden fixture from the contract tests.",
    _validation_pair["path"],
    _validation_pair["request"],
)

example(
    "Infeasible request (422)",
    "A domain error passed through verbatim from the library — Sobol requires "
    "power-of-two run counts, so it names the nearest valid sizes.",
    "/v1/designs/space-filling",
    {"factors": TEMP_TIME, "sampler": "sobol", "n_runs": 10},
)


# --------------------------------------------------------------------------- #
# Write it out
# --------------------------------------------------------------------------- #
OUT.write_text("\n".join(blocks).rstrip() + "\n")
print(f"wrote {OUT} ({len(blocks)} blocks)")
