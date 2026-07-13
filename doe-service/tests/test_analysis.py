"""HTTP-level tests for the Milestone 3 analysis router (``docs/WEBSERVICE_BUILD.md`` §3).

Anchored the same way the library anchors known designs (``CLAUDE.md``) and the M2
designs-router tests (``test_designs.py``): golden designs built directly with ``doe``,
POSTed through the real HTTP layer, and (where the plan calls for it) checked for exact
equality against the library's own ``to_dict()`` output -- proving the service adds
nothing beyond ``warnings``.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from doe import (
    ContinuousFactor,
    Design,
    FactorSet,
    MixtureFactor,
    fit_ols,
    full_factorial,
)
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _yield_factors() -> FactorSet:
    return FactorSet(
        [
            ContinuousFactor("temperature", low=40.0, high=80.0),
            ContinuousFactor("time", low=5.0, high=15.0),
        ]
    )


def _yield_design() -> Design:
    """The 2^2 full-factorial from the ``fit_ols``/``FitResult.to_dict`` doctests."""
    design = full_factorial(_yield_factors())
    coded = design.coded()
    response = 50 + 3 * coded["temperature"] + 2 * coded["time"]
    return design.with_response("yield", response)


def _mixture_design() -> Design:
    """The two-component Scheffé blend from the ``fit_ols`` doctest."""
    factors = FactorSet([MixtureFactor("A"), MixtureFactor("B")])
    blend = Design(
        pd.DataFrame({"A": [1.0, 0.0, 0.5], "B": [0.0, 1.0, 0.5]}),
        factors,
    )
    return blend.with_response("yield", [12.0, 18.0, 15.0])


# --------------------------------------------------------------------------- #
# /fit
# --------------------------------------------------------------------------- #


def test_fit_response_equals_library_to_dict(client: TestClient) -> None:
    """The service adds nothing to a non-saturated fit but the ``warnings`` array."""
    design = _yield_design()
    model = {"order": 1, "interactions": False}
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": model,
    }

    response = client.post("/v1/analysis/fit", json=body)
    assert response.status_code == 200, response.text

    expected = fit_ols(design, "yield", order=1, interactions=False).to_dict()
    assert response.json() == {**expected, "warnings": []}


def test_fit_saturated_model_has_null_std_errors_and_warns(client: TestClient) -> None:
    """A 2x2 full factorial with intercept + both main effects + interaction (4 terms,
    4 runs) is saturated under the ``"linear"`` default (order=1, interactions=True)."""
    design = _yield_design()
    body = {"design": design.to_dict(), "response": "yield"}

    response = client.post("/v1/analysis/fit", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["dof_resid"] == 0
    assert all(term["std_error"] is None for term in data["terms"])
    assert "saturated_model" in data["warnings"]


def test_fit_scheffe_model_has_null_effects(client: TestClient) -> None:
    design = _mixture_design()
    body = {"design": design.to_dict(), "response": "yield", "model": "scheffe-linear"}

    response = client.post("/v1/analysis/fit", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["terms"], "expected at least one term"
    assert all(term["effect"] is None for term in data["terms"])


def test_fit_response_typo_returns_422_listing_available_columns(client: TestClient) -> None:
    design = _yield_design()
    body = {"design": design.to_dict(), "response": "yeild"}  # typo

    response = client.post("/v1/analysis/fit", json=body)
    assert response.status_code == 422, response.text

    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any("yield" in e for e in error["errors"])
    assert any("yeild" in e for e in error["errors"])


# --------------------------------------------------------------------------- #
# /anova
# --------------------------------------------------------------------------- #


def test_anova_without_replicates_has_null_lack_of_fit_and_warns(client: TestClient) -> None:
    design = _yield_design()
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": {"order": 1, "interactions": False},
    }

    response = client.post("/v1/analysis/anova", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["lack_of_fit"] is None
    assert "no_pure_error" in data["warnings"]
    # dof_resid = 4 - 3 = 1 here, so press/predicted_r2 *are* defined.
    assert data["press"] is not None
    assert data["predicted_r2"] is not None
    assert len(data["rows"]) > 0


def test_anova_saturated_fit_has_null_press_and_predicted_r2(client: TestClient) -> None:
    design = _yield_design()
    body = {"design": design.to_dict(), "response": "yield"}  # default: saturated

    response = client.post("/v1/analysis/anova", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["press"] is None
    assert data["predicted_r2"] is None


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


def test_predict_matches_fit_predictions(client: TestClient) -> None:
    design = _yield_design()
    fit = fit_ols(design, "yield", order=1, interactions=False)
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": {"order": 1, "interactions": False},
        "points": [{"temperature": 40, "time": 5}, {"temperature": 80, "time": 15}],
    }

    response = client.post("/v1/analysis/predict", json=body)
    assert response.status_code == 200, response.text

    expected = fit.predict(
        pd.DataFrame({"temperature": [40, 80], "time": [5, 15]})
    ).tolist()
    assert response.json()["predictions"] == pytest.approx(expected)


def test_predict_missing_factor_names_it_and_the_record_index(client: TestClient) -> None:
    design = _yield_design()
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "points": [
            {"temperature": 40, "time": 5},
            {"temperature": 80},  # missing "time"
        ],
    }

    response = client.post("/v1/analysis/predict", json=body)
    assert response.status_code == 422, response.text

    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any("time" in e and "1" in e for e in error["errors"])


# --------------------------------------------------------------------------- #
# /diagnostics
# --------------------------------------------------------------------------- #


def test_diagnostics_reports_efficiency_vif_condition_number_correlation_leverage(
    client: TestClient,
) -> None:
    design = full_factorial(_yield_factors())
    body = {"design": design.to_dict()}

    response = client.post("/v1/analysis/diagnostics", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert set(data["efficiency"]) == {"d", "a", "g", "i"}
    assert data["efficiency"]["d"] == pytest.approx(1.0)  # full factorial is D-optimal
    assert data["condition_number"] == pytest.approx(1.0)
    # default model is {"order": 1, "interactions": true}, so the interaction term rides
    # along too.
    assert set(data["vif"]) == {"temperature", "time", "temperature:time"}
    assert data["correlation_matrix"]["labels"]
    assert len(data["leverage"]) == len(design.runs)


# --------------------------------------------------------------------------- #
# /coverage
# --------------------------------------------------------------------------- #


def test_coverage_reports_discrepancy_and_maximin_distance(client: TestClient) -> None:
    design = full_factorial(_yield_factors())
    body = {"design": design.to_dict(), "method": "CD"}

    response = client.post("/v1/analysis/coverage", json=body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["discrepancy"] >= 0
    assert data["maximin_distance"] >= 0


# --------------------------------------------------------------------------- #
# NaN / Infinity never appear as bare literals on the wire
# --------------------------------------------------------------------------- #


def test_no_response_body_contains_a_bare_nan_or_infinity_literal(client: TestClient) -> None:
    design = _yield_design()

    responses = [
        client.post("/v1/analysis/fit", json={"design": design.to_dict(), "response": "yield"}),
        client.post("/v1/analysis/anova", json={"design": design.to_dict(), "response": "yield"}),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        text = response.text
        assert "NaN" not in text
        assert "Infinity" not in text
        # round-trips through the stdlib decoder without needing NaN/Infinity constants
        json.loads(text)


def test_fit_response_json_has_no_extra_fields(client: TestClient) -> None:
    """Sanity check that the response model does not leak internal fields."""
    design = _yield_design()
    body = {
        "design": design.to_dict(),
        "response": "yield",
        "model": {"order": 1, "interactions": False},
    }
    response = client.post("/v1/analysis/fit", json=body)
    data: dict[str, Any] = response.json()
    assert set(data) == {
        "terms",
        "r_squared",
        "adjusted_r2",
        "dof_resid",
        "mse",
        "fitted",
        "residuals",
        "model",
        "warnings",
    }
