"""HTTP-level tests for the ``doe-web`` app factory.

Covers the two things ``create_app`` wires together (see ``src/doe_web/main.py``): the
static single-page UI served from ``/``, and the full doe-service API mounted unmodified
under ``/api``. The end-to-end flow test replays exactly the request sequence the UI's
``app.js`` makes -- generate a design, attach a response, fit it, then ask for a surface
plot grid -- proving the mount is wired correctly rather than just "some route answers".
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from doe_web.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# static single-page UI at `/`
# --------------------------------------------------------------------------- #


def test_root_serves_the_spa(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "DoE planner" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    js = client.get("/app.js")
    css = client.get("/style.css")
    assert js.status_code == 200
    assert css.status_code == 200


# --------------------------------------------------------------------------- #
# doe-service mounted under `/api`
# --------------------------------------------------------------------------- #


def test_doe_service_health_is_mounted_under_api(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["doe_version"]


def test_large_optimal_search_runs_in_parallel(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """doe-web mounts the service with parallelism enabled: a large D-optimal search fans its
    restarts across cores (``n_jobs == -1``), while a small one stays single-process."""
    import doe_service.routers.designs as designs_router
    from doe_web.main import OPTIMAL_PARALLEL_MIN_RUNS

    seen: dict[str, Any] = {}
    original = designs_router._coordinate_exchange

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["n_jobs"] = kwargs.get("n_jobs")
        return original(*args, **kwargs)

    monkeypatch.setattr(designs_router, "_coordinate_exchange", spy)

    factors = [
        {"type": "continuous", "name": n, "low": 0, "high": 1} for n in ("a", "b", "c", "d")
    ]

    small = client.post(
        "/api/v1/designs/optimal",
        json={"factors": factors, "n_runs": OPTIMAL_PARALLEL_MIN_RUNS - 1, "seed": 0},
    )
    assert small.status_code == 200, small.text
    assert seen["n_jobs"] == 1

    large = client.post(
        "/api/v1/designs/optimal",
        json={"factors": factors, "n_runs": OPTIMAL_PARALLEL_MIN_RUNS + 6, "seed": 0},
    )
    assert large.status_code == 200, large.text
    assert seen["n_jobs"] == -1


def test_end_to_end_design_response_fit_surface_flow(client: TestClient) -> None:
    """Replays the UI's request sequence: generate -> attach responses -> fit -> plot."""
    factors = [
        {"type": "continuous", "name": "temp", "low": 20, "high": 80, "units": "C"},
        {"type": "continuous", "name": "time", "low": 0, "high": 10, "units": None},
    ]

    design_response = client.post(
        "/api/v1/designs/central-composite", json={"factors": factors}
    )
    assert design_response.status_code == 200, design_response.text
    design_body: dict[str, Any] = design_response.json()
    runs = design_body["design"]["runs"]
    assert isinstance(runs, list)
    assert len(runs) > 0

    yields = [float(i) for i in range(len(runs))]
    responses_response = client.post(
        "/api/v1/designs/responses",
        json={"design": design_body["design"], "responses": {"yield": yields}},
    )
    assert responses_response.status_code == 200, responses_response.text
    responses_body: dict[str, Any] = responses_response.json()
    responses_runs = responses_body["design"]["runs"]
    assert all("yield" in run for run in responses_runs)
    assert [run["yield"] for run in responses_runs] == yields

    fit_response = client.post(
        "/api/v1/analysis/fit",
        json={
            "design": responses_body["design"],
            "response": "yield",
            "model": "quadratic",
        },
    )
    assert fit_response.status_code == 200, fit_response.text
    fit_body: dict[str, Any] = fit_response.json()
    assert "r_squared" in fit_body
    assert len(fit_body["terms"]) > 0

    # The model-adequacy panel draws residuals-vs-fitted and normal-Q-Q from these two
    # fields, so the fit response must carry them (one per run).
    assert len(fit_body["fitted"]) == len(runs)
    assert len(fit_body["residuals"]) == len(runs)

    # ... and it fetches the lack-of-fit test + predicted-R2 from /analysis/anova. A CCD has
    # replicated centre points, so lack-of-fit is estimable (not null).
    anova_response = client.post(
        "/api/v1/analysis/anova",
        json={
            "design": responses_body["design"],
            "response": "yield",
            "model": "quadratic",
        },
    )
    assert anova_response.status_code == 200, anova_response.text
    anova_body: dict[str, Any] = anova_response.json()
    assert "predicted_r2" in anova_body
    assert anova_body["lack_of_fit"] is not None
    assert "p" in anova_body["lack_of_fit"]

    surface_response = client.post(
        "/api/v1/plot-data/surface",
        json={
            "design": responses_body["design"],
            "response": "yield",
            "model": "quadratic",
            "x": "temp",
            "y": "time",
            "resolution": 10,
        },
    )
    assert surface_response.status_code == 200, surface_response.text
    surface_body: dict[str, Any] = surface_response.json()
    z = surface_body["z"]
    assert len(z) == 10
    assert all(len(row) == 10 for row in z)


def test_screening_flow_exposes_generator_meta_and_effects(client: TestClient) -> None:
    """The screening results view (a 2-level factorial) is chosen from the generator recorded on
    a loaded plan and drawn from the fit's per-term effects, so both must be on the wire: the
    design carries ``meta.generator.name == "full_factorial"`` with ``parameters.levels`` all 2
    (one count per factor), and every fitted term reports an ``effect`` (the ±1 coded swing the
    Pareto / half-normal use)."""
    factors = [
        {"type": "continuous", "name": n, "low": 0, "high": 1} for n in ("a", "b", "c")
    ]

    design_response = client.post(
        "/api/v1/designs/full-factorial", json={"factors": factors, "levels": 2}
    )
    assert design_response.status_code == 200, design_response.text
    design = design_response.json()["design"]

    generator = design["meta"]["generator"]
    assert generator["name"] == "full_factorial"
    levels = generator["parameters"]["levels"]
    # A per-factor list (the JS screening detection accepts either an all-2 list or scalar 2).
    assert levels == [2, 2, 2]

    responses = [float(i % 3) for i in range(len(design["runs"]))]
    design = client.post(
        "/api/v1/designs/responses",
        json={"design": design, "responses": {"y": responses}},
    ).json()["design"]

    fit_response = client.post(
        "/api/v1/analysis/fit",
        json={"design": design, "response": "y", "model": "quadratic"},
    )
    assert fit_response.status_code == 200, fit_response.text
    non_intercept = [t for t in fit_response.json()["terms"] if t["term"] != "Intercept"]
    assert non_intercept
    assert all("effect" in t for t in non_intercept)


def test_plackett_burman_design_for_six_factors(client: TestClient) -> None:
    """The quick-screen reroute (app.js generatePlan(), > 4 factors) posts here instead of
    /designs/full-factorial. 6 factors need the smallest constructible Plackett-Burman size
    with n >= k + 1 -- that's 8 runs (the 2-level full factorial would need 2**6 = 64)."""
    factors = [
        {"type": "continuous", "name": n, "low": 0, "high": 1}
        for n in ("a", "b", "c", "d", "e", "f")
    ]

    response = client.post("/api/v1/designs/plackett-burman", json={"factors": factors})
    assert response.status_code == 200, response.text
    design = response.json()["design"]
    assert len(design["runs"]) == 8
    assert design["meta"]["generator"]["name"] == "plackett_burman"
    # Every factor column only takes its two coded extremes.
    for f in factors:
        values = {run[f["name"]] for run in design["runs"]}
        assert values == {0.0, 1.0}


def test_plackett_burman_fit_uses_main_effects_only_model(client: TestClient) -> None:
    """app.js's fitRequest() sends {"order": 1, "interactions": false} for a Plackett-Burman
    plan (its recorded generator, per isScreeningPlan()'s SCREENING_GENERATORS set) rather
    than the default "quadratic" model: 6 factors in 8 runs gives 22 quadratic terms
    (intercept + 6 main effects + 15 pairwise interactions) but only 7 main-effects terms,
    so "quadratic" would be rank-deficient. The main-effects-only fit must instead succeed
    with an estimable p-value on every term."""
    factors = [
        {"type": "continuous", "name": n, "low": 0, "high": 1}
        for n in ("a", "b", "c", "d", "e", "f")
    ]
    design = client.post(
        "/api/v1/designs/plackett-burman", json={"factors": factors}
    ).json()["design"]

    responses = [float(i) for i in range(len(design["runs"]))]
    design = client.post(
        "/api/v1/designs/responses",
        json={"design": design, "responses": {"yield": responses}},
    ).json()["design"]

    # The quadratic model the UI uses everywhere else has 22 terms (intercept + 6 main
    # effects + 15 pairwise interactions) but only 8 runs -- fewer runs than terms means the
    # model matrix cannot be full rank, so the library refuses to fit it at all (a 422, not
    # a silently-wrong answer). This is exactly the broken state the reroute must avoid.
    quadratic_fit = client.post(
        "/api/v1/analysis/fit",
        json={"design": design, "response": "yield", "model": "quadratic"},
    )
    assert quadratic_fit.status_code == 422, quadratic_fit.text

    fit_response = client.post(
        "/api/v1/analysis/fit",
        json={
            "design": design,
            "response": "yield",
            "model": {"order": 1, "interactions": False},
        },
    )
    assert fit_response.status_code == 200, fit_response.text
    body = fit_response.json()
    terms = [t for t in body["terms"] if t["term"] != "Intercept"]
    assert len(terms) == 6  # one main effect per factor, no interactions
    assert all(t["effect"] is not None for t in terms)
    assert any(t["p"] is not None for t in terms)


def test_run_count_formulas_anchor_client_side_preview(client: TestClient) -> None:
    """app.js shows a live, client-side run-count preview in step 1 (onFactorsChanged()'s
    centralCompositeRunCount/boxBehnkenRunCount/fullFactorialRunCount/pbRunCount helpers) so a
    user sees the bench cost before clicking "Create experimental plan". These assertions
    anchor those formulas against what the service actually generates, so the two can't drift
    apart: central_composite's 2^k full-factorial core (the UI never passes a `fraction`) plus
    2k axial runs plus the service's default center=4; box_behnken's 4 * C(k, 2) edge runs
    (one run per +/-1 x +/-1 combination of each factor pair) plus the default center=3; and
    the quick-screen counts (full-factorial levels=2, and its > 4 factor Plackett-Burman
    reroute) already exercised by test_plackett_burman_design_for_six_factors above."""

    def continuous_factors(k: int) -> list[dict[str, Any]]:
        return [{"type": "continuous", "name": f"f{i}", "low": 0, "high": 1} for i in range(k)]

    # central_composite: 2**k + 2*k + 4 (k=2 -> 4+4+4=12, k=3 -> 8+6+4=18, k=4 -> 16+8+4=28).
    for k, expected in [(2, 12), (3, 18), (4, 28)]:
        response = client.post(
            "/api/v1/designs/central-composite", json={"factors": continuous_factors(k)}
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["design"]["runs"]) == expected

    # box_behnken: 4 * C(k, 2) + 3 (k=3 -> 12+3=15, k=4 -> 24+3=27).
    for k, expected in [(3, 15), (4, 27)]:
        response = client.post(
            "/api/v1/designs/box-behnken", json={"factors": continuous_factors(k)}
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["design"]["runs"]) == expected

    # full-factorial at levels=2, k=3: 2**3 = 8 runs.
    ff_response = client.post(
        "/api/v1/designs/full-factorial",
        json={"factors": continuous_factors(3), "levels": 2},
    )
    assert ff_response.status_code == 200, ff_response.text
    assert len(ff_response.json()["design"]["runs"]) == 8

    # plackett_burman, k=6: the smallest constructible size with n >= k + 1 is 8 runs (also
    # covered above by test_plackett_burman_design_for_six_factors).
    pb_response = client.post(
        "/api/v1/designs/plackett-burman", json={"factors": continuous_factors(6)}
    )
    assert pb_response.status_code == 200, pb_response.text
    assert len(pb_response.json()["design"]["runs"]) == 8


def test_categorical_dopt_flow_generate_fit_surface(client: TestClient) -> None:
    """Replays the UI's categorical path: a D-optimal design over mixed factors ->
    attach responses -> fit (deviation-coded categorical term) -> surface over the two
    continuous axes with the categorical held at a level -> best settings via
    /optimize/categorical-optimum.

    The plain surface optimizer (/optimize/optimum) has no coded box for a categorical
    factor and returns 422; the UI does not call it for a categorical fit. Instead it calls
    /optimize/categorical-optimum, which optimizes the continuous factors exactly within
    each categorical-level combination and names the winning level -- both contracts are
    guarded here."""
    factors = [
        {"type": "continuous", "name": "temp", "low": 20, "high": 80, "units": "C"},
        {"type": "continuous", "name": "time", "low": 2, "high": 10, "units": "min"},
        {"type": "categorical", "name": "catalyst", "levels": ["A", "B"], "units": None},
    ]

    design_response = client.post(
        "/api/v1/designs/optimal",
        json={"factors": factors, "n_runs": 16, "model": "quadratic", "seed": 0},
    )
    assert design_response.status_code == 200, design_response.text
    design = design_response.json()["design"]
    assert [f["type"] for f in design["factors"]] == ["continuous", "continuous", "categorical"]

    responses = [float(i % 5) + 60 for i in range(len(design["runs"]))]
    design = client.post(
        "/api/v1/designs/responses",
        json={"design": design, "responses": {"yield": responses}},
    ).json()["design"]

    fit_response = client.post(
        "/api/v1/analysis/fit",
        json={"design": design, "response": "yield", "model": "quadratic"},
    )
    assert fit_response.status_code == 200, fit_response.text
    terms = [t["term"] for t in fit_response.json()["terms"]]
    assert "catalyst[B]" in terms  # categorical expanded by deviation coding

    surface_response = client.post(
        "/api/v1/plot-data/surface",
        json={
            "design": design,
            "response": "yield",
            "model": "quadratic",
            "x": "temp",
            "y": "time",
            "resolution": 10,
            "fixed": {"catalyst": "A"},
        },
    )
    assert surface_response.status_code == 200, surface_response.text
    assert len(surface_response.json()["z"]) == 10

    # Surface optimization has no coded box for a categorical factor -> 422 (the UI skips it).
    optimum_response = client.post(
        "/api/v1/optimize/optimum",
        json={"design": design, "response": "yield", "model": "quadratic", "maximize": True},
    )
    assert optimum_response.status_code == 422

    # ... so the UI finds the best settings via the mixed optimizer instead.
    best_response = client.post(
        "/api/v1/optimize/categorical-optimum",
        json={"design": design, "response": "yield", "model": "quadratic", "maximize": True},
    )
    assert best_response.status_code == 200, best_response.text
    best = best_response.json()
    assert set(best["settings"]) == {"temp", "time", "catalyst"}
    assert best["settings"]["catalyst"] in ("A", "B")
    assert best["levels"] == {"catalyst": best["settings"]["catalyst"]}
