"""FactorSet cross-factor invariants reach the wire as 422, not 500 (code review 2026-07-16, #1).

A design document that is well-formed per factor but violates a ``FactorSet`` rule --
mixture components mixed with continuous/categorical factors, a lone mixture component,
or an infeasible mixture blend -- used to pass ``validate_design_dict`` and then raise a
bare ``ValueError`` from ``Design.from_dict``, surfacing as a 500 ``internal``.
``validate_design_dict`` now enforces those rules, so every design-consuming endpoint
answers with an actionable 422 ``validation_error`` and ``/validate`` reports the document
invalid instead of ``{valid: true}``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _mixture_plus_continuous() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "factors": [
            {"type": "mixture", "name": "A", "low": 0.0, "high": 1.0},
            {"type": "continuous", "name": "T", "low": 20.0, "high": 80.0},
        ],
        "runs": [{"A": 0.5, "T": 50.0}],
    }


def _infeasible_blend() -> dict[str, Any]:
    # sum(low) = 1.2 > 1, so no blend can sum to 1 -- an infeasible mixture region
    return {
        "schema_version": "1.0",
        "factors": [
            {"type": "mixture", "name": "A", "low": 0.6, "high": 0.9},
            {"type": "mixture", "name": "B", "low": 0.6, "high": 0.9},
        ],
        "runs": [{"A": 0.5, "B": 0.5}],
    }


def _lone_mixture() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "factors": [{"type": "mixture", "name": "A", "low": 0.0, "high": 1.0}],
        "runs": [{"A": 1.0}],
    }


_CASES = [
    (_mixture_plus_continuous(), "combined with other factor types"),
    (_infeasible_blend(), "feasible blend"),
    (_lone_mixture(), "at least 2 components"),
]


@pytest.mark.parametrize("document,needle", _CASES)
def test_infeasible_factor_set_on_a_design_endpoint_is_422_not_500(
    client: TestClient, document: dict[str, Any], needle: str
) -> None:
    # /analysis/diagnostics is one of the many endpoints that funnel through
    # design_from_document; the review reproduced the 500 here and on /analysis/coverage.
    response = client.post("/v1/analysis/diagnostics", json={"design": document})
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any(needle in message for message in error["errors"]), error["errors"]


@pytest.mark.parametrize("document,needle", _CASES)
def test_validate_reports_infeasible_factor_set_as_invalid(
    client: TestClient, document: dict[str, Any], needle: str
) -> None:
    response = client.post("/v1/designs/validate", json={"design": document})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False
    assert any(needle in message for message in body["errors"]), body["errors"]
