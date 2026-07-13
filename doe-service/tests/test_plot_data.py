"""HTTP-level tests for the Milestone 5 plot-data router (``docs/WEBSERVICE_BUILD.md`` §5).

Anchored the same way the M3/M4 tests are: golden designs built directly with ``doe``,
POSTed through the real HTTP layer, and (for ``/surface``) checked for exact equality
against the library's own headless core -- proving the service only JSON-encodes. The
error cases exercise the router's precondition guards, which convert the cores' ``KeyError``/
``TypeError`` (invisible to ``call_library``) into 422s rather than 500s.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from doe import (
    CategoricalFactor,
    ContinuousFactor,
    Design,
    FactorSet,
    MixtureFactor,
    central_composite,
    simplex_lattice,
)
from doe.analysis.fit import fit_ols
from doe.plotting import surface_grid
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def safe_client() -> TestClient:
    # raise_server_exceptions=False lets a 500 be observed as a response rather than
    # re-raised, so "must not be a 500" is a real assertion.
    return TestClient(create_app(), raise_server_exceptions=False)


def _surface_design() -> Design:
    """A rotatable CCD in two continuous factors with a genuine quadratic response."""
    factors = FactorSet(
        [
            ContinuousFactor("temperature", low=40.0, high=80.0),
            ContinuousFactor("time", low=5.0, high=15.0),
        ]
    )
    design = central_composite(factors, alpha="rotatable", center=4)
    coded = design.coded()
    response = (
        60.0
        + 4.0 * coded["temperature"]
        + 2.0 * coded["time"]
        - 3.0 * coded["temperature"] ** 2
        - 2.0 * coded["time"] ** 2
        + 1.5 * coded["temperature"] * coded["time"]
    )
    return design.with_response("yield", response)


def _mixture_design() -> Design:
    """A 3-component simplex-lattice blend for the ternary surface."""
    factors = FactorSet([MixtureFactor("A"), MixtureFactor("B"), MixtureFactor("C")])
    design = simplex_lattice(factors, degree=2)
    coded = design.coded()
    response = 10.0 * coded["A"] + 12.0 * coded["B"] + 8.0 * coded["C"]
    return design.with_response("yield", response)


def _quad_body(design: Design) -> dict[str, object]:
    return {"design": design.to_dict(), "response": "yield", "model": "quadratic"}


# --------------------------------------------------------------------------- #
# /surface
# --------------------------------------------------------------------------- #


def test_surface_grid_shape_and_values_match_library(client: TestClient) -> None:
    """The service only JSON-encodes: its mesh equals a direct ``surface_grid`` call."""
    design = _surface_design()
    resolution = 15
    body = {**_quad_body(design), "x": "temperature", "y": "time", "resolution": resolution}
    response = client.post("/v1/plot-data/surface", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()

    for key in ("x", "y", "z"):
        assert np.shape(payload[key]) == (resolution, resolution)

    result = fit_ols(design, "yield", order=2, interactions=True)
    nat_x, nat_y, z = surface_grid(result, "temperature", "time", resolution=resolution)
    assert np.allclose(payload["x"], nat_x)
    assert np.allclose(payload["y"], nat_y)
    assert np.allclose(payload["z"], z)


def test_surface_same_axis_is_infeasible(client: TestClient) -> None:
    design = _surface_design()
    body = {**_quad_body(design), "x": "temperature", "y": "temperature"}
    response = client.post("/v1/plot-data/surface", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


def test_surface_categorical_axis_is_infeasible(safe_client: TestClient) -> None:
    """A categorical axis makes the core raise ``TypeError`` -- must be 422, not 500."""
    factors = FactorSet(
        [
            ContinuousFactor("temperature", low=40.0, high=80.0),
            CategoricalFactor("catalyst", ("acid", "base")),
        ]
    )
    # a tiny hand-built design so the fit resolves; values are immaterial to the guard
    design = Design(
        pd.DataFrame(
            {
                "temperature": [40.0, 80.0, 40.0, 80.0],
                "catalyst": ["acid", "acid", "base", "base"],
            }
        ),
        factors,
    ).with_response("yield", [1.0, 2.0, 3.0, 4.0])
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": {"order": 1, "interactions": True},
        "x": "temperature",
        "y": "catalyst",
    }
    response = safe_client.post("/v1/plot-data/surface", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


def test_surface_unknown_axis_is_validation_error(client: TestClient) -> None:
    design = _surface_design()
    body = {**_quad_body(design), "x": "temperature", "y": "pressure"}
    response = client.post("/v1/plot-data/surface", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_surface_resolution_over_cap_is_limit_exceeded(client: TestClient) -> None:
    design = _surface_design()
    body = {**_quad_body(design), "x": "temperature", "y": "time", "resolution": 10_000}
    response = client.post("/v1/plot-data/surface", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


# --------------------------------------------------------------------------- #
# /interactions
# --------------------------------------------------------------------------- #


def test_interactions_shape(client: TestClient) -> None:
    design = _surface_design()
    resolution = 12
    body = {
        **_quad_body(design),
        "x": "temperature",
        "trace": "time",
        "resolution": resolution,
    }
    response = client.post("/v1/plot-data/interactions", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["x"]) == resolution
    assert len(payload["lines"]) == 2  # default trace_levels: low and high
    for line in payload["lines"]:
        assert len(line["z"]) == resolution
        assert "trace_value" in line


# --------------------------------------------------------------------------- #
# /ternary
# --------------------------------------------------------------------------- #


def test_ternary_on_mixture_fit(client: TestClient) -> None:
    design = _mixture_design()
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": "scheffe-quadratic",
        "resolution": 10,
    }
    response = client.post("/v1/plot-data/ternary", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["x"]) == len(payload["y"]) == len(payload["z"])
    assert all(len(point) == 3 for point in payload["points"])


def test_ternary_on_non_mixture_fit_is_infeasible(client: TestClient) -> None:
    design = _surface_design()
    body = {"design": design.to_dict(), "response": "yield", "model": "quadratic"}
    response = client.post("/v1/plot-data/ternary", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


# --------------------------------------------------------------------------- #
# /alias
# --------------------------------------------------------------------------- #


def test_alias_returns_labels_and_square_matrix(client: TestClient) -> None:
    design = _surface_design()
    body = {"design": design.to_dict(), "model": {"order": 1, "interactions": True}}
    response = client.post("/v1/plot-data/alias", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    n = len(payload["labels"])
    assert n > 0
    assert len(payload["matrix"]) == n
    assert all(len(row) == n for row in payload["matrix"])


# --------------------------------------------------------------------------- #
# cross-cutting
# --------------------------------------------------------------------------- #


def test_no_plot_route_returns_500_on_documented_errors(safe_client: TestClient) -> None:
    """Every documented failure input must be a 4xx envelope, never a 500."""
    design = _surface_design()
    bad_requests = [
        ("/v1/plot-data/surface", {**_quad_body(design), "x": "temperature", "y": "temperature"}),
        ("/v1/plot-data/surface", {**_quad_body(design), "x": "temperature", "y": "pressure"}),
        (
            "/v1/plot-data/ternary",
            {"design": design.to_dict(), "response": "yield", "model": "quadratic"},
        ),
    ]
    for path, body in bad_requests:
        response = safe_client.post(path, json=body)
        assert response.status_code != 500, f"{path}: {response.text}"
        assert response.status_code == 422, f"{path}: {response.status_code}"


def test_no_response_body_contains_nan_literal(client: TestClient) -> None:
    design = _surface_design()
    responses = [
        client.post(
            "/v1/plot-data/surface",
            json={**_quad_body(design), "x": "temperature", "y": "time", "resolution": 8},
        ),
        client.post(
            "/v1/plot-data/interactions",
            json={**_quad_body(design), "x": "temperature", "trace": "time", "resolution": 8},
        ),
        client.post(
            "/v1/plot-data/alias",
            json={"design": design.to_dict(), "model": {"order": 1, "interactions": True}},
        ),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        assert "NaN" not in response.text
        assert "Infinity" not in response.text
        json.loads(response.text)
