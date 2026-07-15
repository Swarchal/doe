# doe-web

Browser front-end prototype over `doe-service` — a guided experimental-plan builder for non-technical users. Define factors → generate a design → enter measurements → fit/optimum/contour map, all without code.

## Architecture

`doe-web` is a uv-workspace member. One FastAPI process mounts the full `doe-service` API under `/api` and serves a static single-page UI (vanilla JavaScript) from `/`. Same origin, so no CORS configuration needed.

The UI walks through three steps:
1. **Define factors**: name each variable and choose its type — a **continuous** factor with low/high bounds, or a **categorical** factor with a set of named options — then choose a plan type (central composite, Box–Behnken, 2-level full factorial, or a D-optimal custom design that supports categorical factors).
2. **Run experiments**: download a randomized run sheet (CSV), enter your measurements, optionally fill with demo results to see the full flow.
3. **Analyse**: inspect model fit (R²), best predicted settings, factor effects, and an interactive response-surface contour map. With a categorical factor, the contour maps two continuous axes with each categorical factor held at a chosen level; the best settings still appear — since surface optimization has no coded box for a categorical factor, the UI calls `/api/v1/optimize/categorical-optimum`, which optimizes the continuous factors exactly within each combination of categorical levels and reports the winning combination (level names included).

State is a design document held in the browser; the service stays stateless. **Save plan (JSON)** downloads that document (the `Design.to_dict()` wire format, with any measurements typed so far folded in), and **Load saved plan (JSON)** reads one back — validated through `POST /api/v1/designs/validate` — to pick up where you left off, so a study survives a closed tab and round-trips with any other `doe` tooling.

## Running locally

From this directory:

```bash
uv run --extra dev uvicorn --factory doe_web.main:create_app --reload
```

Then open **http://127.0.0.1:8000**. Interactive API docs are at `/api/docs`.

## Checks

```bash
uv run --extra dev pytest        # run the test suite
uv run --extra dev mypy          # strict type-checking (see pyproject.toml for config)
```

## Notes

- **Continuous and categorical factors.** Central composite / Box–Behnken need all-continuous factors; the 2-level full factorial and the D-optimal custom design both accept categorical factors.
- **Parallel D-optimal search for large plans.** The D-optimal / augment coordinate-exchange search runs its independent restarts across CPU cores when the design is large (≥ `OPTIMAL_PARALLEL_MIN_RUNS` runs, see `main.py`); smaller plans stay single-process, where the worker start-up overhead would outweigh the gain. This is a server-side decision — `doe-web` mounts the API with a parallel-enabled `Limits`; the client never sets it, and the plain `doe-service` leaves it disabled. See `doe.coordinate_exchange`'s `n_jobs` and `doe_service.limits.Limits.optimal_n_jobs`.
- **Four plan types wired** (central composite, Box–Behnken, 2-level full factorial, D-optimal). More generators can be added as UI panels.
- **Plotly.js from CDN** for the contour map; the page degrades gracefully offline (map will not render, but the form and table remain).
- **Demo results button** generates a deterministic synthetic response so the full analysis flow can be demoed without lab data.
