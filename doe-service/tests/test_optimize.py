"""HTTP-level tests for the Milestone 4 optimize router (``docs/WEBSERVICE_BUILD.md`` §4).

Anchored the same way the other router tests are anchored (``CLAUDE.md``,
``test_analysis.py``): golden designs built directly with ``doe``, POSTed through the
real HTTP layer. Two anchors are the walkthrough docs themselves --
``docs/WORKFLOW.md``'s known optimum and ``docs/WORKFLOW2.md``'s known desirability
balance -- reproduced exactly (same factors, same seeds) per
``scripts/build_workflow_assets.py``/``scripts/build_workflow2_assets.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from doe import ContinuousFactor, Design, FactorSet, central_composite, full_factorial
from doe_service.limits import DEFAULT_LIMITS
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# WORKFLOW.md anchor: single-response quadratic optimum
# --------------------------------------------------------------------------- #


def _workflow_design() -> Design:
    """The exact WORKFLOW.md design + synthetic yield response (same factors, seeds)."""
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
    return design.with_response("yield_pct", yield_pct)


def test_optimum_reproduces_workflow_md_operating_point(client: TestClient) -> None:
    """``docs/WORKFLOW.md`` #6: Optimum(max: temperature=69.05, time=51.06,
    catalyst=1.779 -> yield_pct=81.53)."""
    design = _workflow_design()
    body = {"design": design.to_dict(), "response": "yield_pct", "model": "quadratic"}

    response = client.post("/v1/optimize/optimum", json=body)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["natural"]["temperature"] == pytest.approx(69.05, abs=0.01)
    assert data["natural"]["time"] == pytest.approx(51.06, abs=0.01)
    assert data["natural"]["catalyst"] == pytest.approx(1.779, abs=0.001)
    assert data["response"] == pytest.approx(81.53, abs=0.01)
    assert data["maximize"] is True
    assert data["at_bound"] is False
    assert data["response_name"] == "yield_pct"
    assert data["warnings"] == []


def test_stationary_point_reproduces_workflow_md_operating_point(client: TestClient) -> None:
    """``docs/WORKFLOW.md`` #6: StationaryPoint(maximum: temperature=69.05, time=51.06,
    catalyst=1.779 -> yield_pct=81.53) -- the peak lands inside the tested box, so this
    agrees with ``/optimum`` exactly."""
    design = _workflow_design()
    body = {"design": design.to_dict(), "response": "yield_pct", "model": "quadratic"}

    response = client.post("/v1/optimize/stationary-point", json=body)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["kind"] == "maximum"
    assert data["natural"]["temperature"] == pytest.approx(69.05, abs=0.01)
    assert data["natural"]["time"] == pytest.approx(51.06, abs=0.01)
    assert data["natural"]["catalyst"] == pytest.approx(1.779, abs=0.001)
    assert data["response"] == pytest.approx(81.53, abs=0.01)
    assert data["response_name"] == "yield_pct"
    assert len(data["eigenvalues"]) == 3
    assert len(data["eigenvectors"]) == 3


# --------------------------------------------------------------------------- #
# WORKFLOW2.md anchor: two-response desirability balance
# --------------------------------------------------------------------------- #


def _workflow2_design() -> Design:
    """The exact WORKFLOW2.md design + synthetic yield/impurity responses."""
    factors = [
        ContinuousFactor("temperature", low=60, high=100, units="C"),
        ContinuousFactor("time", low=30, high=90, units="min"),
    ]
    design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260708)
    coded = design.coded()
    t, m = coded["temperature"], coded["time"]
    rng = np.random.default_rng(2026)
    yield_pct = (
        82.0
        + 6.0 * t
        + 4.5 * m
        - 4.0 * t**2
        - 3.0 * m**2
        + 1.5 * t * m
        + rng.normal(0, 0.7, design.n_runs)
    )
    impurity_pct = (
        8.0 + 3.5 * t + 2.5 * m + 0.8 * t**2 + 0.5 * m**2 + rng.normal(0, 0.4, design.n_runs)
    )
    return design.with_responses(yield_pct=yield_pct, impurity_pct=impurity_pct)


def test_desirability_reproduces_workflow2_md_balance(client: TestClient) -> None:
    """``docs/WORKFLOW2.md`` #6: DesirabilityResult(D=0.4892: temperature=81.32,
    time=62.72 | yield_pct=82.98, impurity_pct=8.635)."""
    design = _workflow2_design()
    body = {
        "design": design.to_dict(),
        "goals": [
            {
                "response": "yield_pct",
                "model": "quadratic",
                "goal": "max",
                "low": 78.0,
                "high": 88.0,
            },
            {
                "response": "impurity_pct",
                "model": "quadratic",
                "goal": "min",
                "low": 5.0,
                "high": 12.0,
            },
        ],
    }

    response = client.post("/v1/optimize/desirability", json=body)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["natural"]["temperature"] == pytest.approx(81.32, abs=0.01)
    assert data["natural"]["time"] == pytest.approx(62.72, abs=0.01)
    assert data["overall"] == pytest.approx(0.4892, abs=0.001)
    assert data["responses"]["yield_pct"] == pytest.approx(82.98, abs=0.01)
    assert data["responses"]["impurity_pct"] == pytest.approx(8.635, abs=0.005)
    assert data["individual"]["yield_pct"] == pytest.approx(0.498, abs=0.001)
    assert data["individual"]["impurity_pct"] == pytest.approx(0.481, abs=0.001)
    assert data["warnings"] == []


# --------------------------------------------------------------------------- #
# error cases
# --------------------------------------------------------------------------- #


def _simple_design() -> Design:
    factors = FactorSet(
        [
            ContinuousFactor("temperature", low=40.0, high=80.0),
            ContinuousFactor("time", low=5.0, high=15.0),
        ]
    )
    design = full_factorial(factors)
    coded = design.coded()
    response = 50 + 3 * coded["temperature"] + 2 * coded["time"]
    return design.with_response("yield", response)


def test_desirability_target_outside_bounds_is_422_infeasible(client: TestClient) -> None:
    design = _simple_design()
    body = {
        "design": design.to_dict(),
        "goals": [
            {
                "response": "yield",
                "model": {"order": 1, "interactions": False},
                "goal": "target",
                "low": 0.0,
                "high": 10.0,
                "target": 100.0,  # outside (low, high)
            }
        ],
    }

    response = client.post("/v1/optimize/desirability", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


def test_optimum_unknown_bound_factor_is_422_validation_error(client: TestClient) -> None:
    design = _workflow_design()
    body = {
        "design": design.to_dict(),
        "response": "yield_pct",
        "model": "quadratic",
        "bounds": {"bogus_factor": [1.0, 2.0]},
    }

    response = client.post("/v1/optimize/optimum", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any("bogus_factor" in e for e in error["errors"])


def test_stationary_point_linear_model_is_422_infeasible(client: TestClient) -> None:
    design = _simple_design()
    body = {"design": design.to_dict(), "response": "yield", "model": "linear"}

    response = client.post("/v1/optimize/stationary-point", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


def test_desirability_too_many_goals_is_422_limit_exceeded(client: TestClient) -> None:
    design = _simple_design()
    goal = {
        "response": "yield",
        "model": {"order": 1, "interactions": False},
        "goal": "max",
        "low": 0.0,
        "high": 100.0,
    }
    body = {
        "design": design.to_dict(),
        "goals": [goal] * (DEFAULT_LIMITS.max_goals + 1),
    }

    response = client.post("/v1/optimize/desirability", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


# --------------------------------------------------------------------------- #
# NaN / Infinity never appear as bare literals on the wire
# --------------------------------------------------------------------------- #


def test_no_response_body_contains_a_bare_nan_or_infinity_literal(client: TestClient) -> None:
    workflow_design = _workflow_design()
    workflow2_design = _workflow2_design()

    responses = [
        client.post(
            "/v1/optimize/optimum",
            json={
                "design": workflow_design.to_dict(),
                "response": "yield_pct",
                "model": "quadratic",
            },
        ),
        client.post(
            "/v1/optimize/stationary-point",
            json={
                "design": workflow_design.to_dict(),
                "response": "yield_pct",
                "model": "quadratic",
            },
        ),
        client.post(
            "/v1/optimize/desirability",
            json={
                "design": workflow2_design.to_dict(),
                "goals": [
                    {
                        "response": "yield_pct",
                        "model": "quadratic",
                        "goal": "max",
                        "low": 78.0,
                        "high": 88.0,
                    },
                    {
                        "response": "impurity_pct",
                        "model": "quadratic",
                        "goal": "min",
                        "low": 5.0,
                        "high": 12.0,
                    },
                ],
            },
        ),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        text = response.text
        assert "NaN" not in text
        assert "Infinity" not in text
        json.loads(text)


def test_mixture_design_is_infeasible_not_a_server_error() -> None:
    """A mixture fit reaches the library's ``TypeError``; it must surface as 422, not 500.

    ``call_library`` only converts ``ValueError``, so without the router's
    ``_require_continuous_factors`` guard every one of these endpoints answers a
    perfectly well-formed request with a 500.
    """
    client = TestClient(create_app(), raise_server_exceptions=False)
    factors = [{"type": "mixture", "name": name} for name in ("x1", "x2", "x3")]
    design = client.post(
        "/v1/designs/simplex-lattice", json={"factors": factors, "degree": 2}
    ).json()["design"]
    for i, run in enumerate(design["runs"]):
        run["y"] = 1.0 + 0.5 * i

    bodies = {
        "/v1/optimize/stationary-point": {
            "design": design,
            "response": "y",
            "model": "scheffe-quadratic",
        },
        "/v1/optimize/optimum": {
            "design": design,
            "response": "y",
            "model": "scheffe-quadratic",
            "maximize": True,
        },
        "/v1/optimize/desirability": {
            "design": design,
            "goals": [
                {
                    "response": "y",
                    "goal": "max",
                    "low": 1.0,
                    "high": 5.0,
                    "model": "scheffe-quadratic",
                }
            ],
        },
    }
    for path, body in bodies.items():
        response = client.post(path, json=body)
        assert response.status_code == 422, f"{path}: {response.status_code} {response.text}"
        assert response.json()["error"]["code"] == "infeasible", path
