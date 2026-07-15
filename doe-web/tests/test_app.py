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
