# doe-web

Browser front-end prototype over `doe-service` — a guided experimental-plan builder for non-technical users. Define factors → generate a design → enter measurements → fit/optimum/contour map, all without code.

## Architecture

`doe-web` is a uv-workspace member. One FastAPI process mounts the full `doe-service` API under `/api` and serves a static single-page UI (vanilla JavaScript) from `/`. Same origin, so no CORS configuration needed.

The UI walks through three steps:
1. **Define factors**: name each variable and choose its type — a **continuous** factor with low/high bounds, or a **categorical** factor with a set of named options — then choose a plan type (central composite, Box–Behnken, 2-level full factorial, or a D-optimal custom design that supports categorical factors); a live "This plan will need *N* runs" preview near the plan-type picker updates as you edit factors or switch plan type, so you know the bench cost before generating. With more than 4 factors, the quick screen automatically switches from the 2-level full factorial (2^k runs — 64 at 6 factors) to a Plackett–Burman design (~k + 1 runs — 8 at 6 factors); a live hint under the plan controls says so. The reroute only applies when every categorical factor has exactly two options (what Plackett–Burman supports) — with a wider categorical factor the plan stays a full factorial and the hint explains why.
2. **Run experiments**: download a randomized run sheet (CSV), then record your measurements — type them in, paste a column copied from a spreadsheet (values spill down the rows), or fill in the run sheet's last column at the bench and re-upload it with **Import results (CSV)**. Optionally fill with demo results to see the full flow.
3. **Analyse**: the results adapt to the kind of plan you built. A **response-surface** plan (central composite, Box–Behnken, D-optimal) shows model fit (R²), best predicted settings, an interactive contour map, factor effects, and a **model-adequacy** panel (predicted R², a plain-language lack-of-fit verdict, and residuals-vs-fitted / normal-Q–Q diagnostic plots — so you can judge whether the fit is trustworthy before acting on the best settings). A **screening** plan (the 2-level "which factors matter?" full factorial, or its Plackett–Burman reroute above 4 factors) instead shows a dedicated **screening view** — an effect-size Pareto chart and a half-normal plot picking out the vital few factors, over the same effects table sorted by importance — since a screen has no curvature to map and no meaningful interior optimum. The screening view is chosen by the plan type when you generate, and inferred from the recorded generator when you load a saved plan. A Plackett–Burman plan is a saturated main-effects design, so its fit uses the main-effects-only model (`{"order": 1, "interactions": false}`) rather than the usual quadratic — the same rank-deficiency reason a Plackett–Burman screen exists in the first place: main effects + every pairwise interaction would need far more runs than it has. With a categorical factor, the contour maps two continuous axes with each categorical factor held at a chosen level; the best settings still appear — since surface optimization has no coded box for a categorical factor, the UI calls `/api/v1/optimize/categorical-optimum`, which optimizes the continuous factors exactly within each combination of categorical levels and reports the winning combination (level names included). Whenever the fit finds a significant interaction between two continuous factors, both views also show an interaction plot — predicted response vs one factor, one line per low/high setting of its partner (a picker appears if there's more than one such pair) — so parallel lines vs. fanning/crossing lines reveal whether the factors act independently or trade off.

State is a design document held in the browser; the service stays stateless. **Save plan (JSON)** downloads that document (the `Design.to_dict()` wire format, with any measurements typed so far folded in), and **Load saved plan (JSON)** reads one back — validated through `POST /api/v1/designs/validate` — to pick up where you left off, so a study survives a closed tab and round-trips with any other `doe` tooling.

## Running locally

From this directory:

```bash
uv run doe-web                # serves http://127.0.0.1:8000
uv run doe-web --reload       # restart on source changes
uv run doe-web --host 0.0.0.0 --port 9000
```

Or from the workspace root: `uv run --package doe-web doe-web`. (The script wraps
`uvicorn --factory doe_web.main:create_app`, which still works directly.)

Then open **http://127.0.0.1:8000**. Interactive API docs are at `/api/docs`.

## Checks

```bash
uv run --extra dev pytest        # run the test suite
uv run --extra dev mypy          # strict type-checking (see pyproject.toml for config)
```

## Notes

- **Results import is forgiving but safe.** The CSV import accepts the downloaded run sheet or a spreadsheet re-save of it (semicolon- or tab-separated variants and comma decimals included), matches rows by the `run` column when present (so a re-sorted sheet still lands on the right runs, falling back to row order), and auto-detects the measurement column. Any factor columns present are cross-checked against the current plan, so a sheet exported from a different or re-generated plan is refused rather than silently attached to the wrong runs. Blank measurement cells stay blank (partial imports are fine). All client-side — the service is untouched.
- **Continuous and categorical factors.** Central composite / Box–Behnken need all-continuous factors; the 2-level full factorial and the D-optimal custom design both accept categorical factors.
- **Parallel D-optimal search for large plans.** The D-optimal / augment coordinate-exchange search runs its independent restarts across CPU cores when the design is large (≥ `OPTIMAL_PARALLEL_MIN_RUNS` runs, see `main.py`); smaller plans stay single-process, where the worker start-up overhead would outweigh the gain. This is a server-side decision — `doe-web` mounts the API with a parallel-enabled `Limits`; the client never sets it, and the plain `doe-service` leaves it disabled. See `doe.coordinate_exchange`'s `n_jobs` and `doe_service.limits.Limits.optimal_n_jobs`.
- **Four plan types wired** (central composite, Box–Behnken, 2-level full factorial, D-optimal). More generators can be added as UI panels. The quick screen has one built-in reroute: above 4 factors it generates a Plackett–Burman design (`/designs/plackett-burman`) instead of the full factorial (`/designs/full-factorial`), and fits it with the main-effects-only model — no dropdown entry for this, it is transparent to the plan-type choice.
- **Plotly.js from CDN** for the contour map; the page degrades gracefully offline (map will not render, but the form and table remain).
- **Demo results button** generates a deterministic synthetic response so the full analysis flow can be demoed without lab data.
- **Debug seed.** The small ⚙ at the right of the load-plan row reveals a "Random seed" field. When set, it is passed to the only two stochastic calls (`/designs/optimal` and `/designs/randomize`), making a generated plan and its run order fully reproducible. Deliberately unobtrusive — it exists for debugging and demos, not for bench users.
