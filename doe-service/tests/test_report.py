"""HTTP-level tests for ``POST /v1/analysis/report`` (the HTML analysis report)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from doe import ContinuousFactor, central_composite, full_factorial
from doe_service.limits import Limits
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _ccd_document() -> dict[str, Any]:
    design = central_composite(
        [ContinuousFactor("temp", 20.0, 80.0), ContinuousFactor("time", 0.0, 10.0)],
        alpha="faced",
    )
    c = design.coded()
    rng = np.random.default_rng(3)
    y = 70 + 2 * c.temp - c.time - 3 * c.temp**2 + rng.normal(0, 0.3, design.n_runs)
    z = 1 + 0.4 * c.time + rng.normal(0, 0.05, design.n_runs)
    return design.with_responses(**{"yield": y, "purity": z}).to_dict()


def test_report_returns_self_contained_html(client: TestClient) -> None:
    response = client.post(
        "/v1/analysis/report",
        json={
            "design": _ccd_document(),
            "responses": [{"response": "yield", "model": "quadratic"}, {"response": "purity"}],
            "title": "Bench study",
            "sections": [
                {
                    "heading": "Best predicted settings",
                    "text": "Maximizing yield.",
                    "rows": [{"temp": 55.1, "time": 3.2, "yield": 72.4}],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Bench study</title>" in html
    assert 'id="yield"' in html and 'id="purity"' in html
    assert "data:image/png;base64," in html
    assert "https://" not in html
    assert "<h2>Best predicted settings</h2>" in html and "Maximizing yield." in html
    assert "<td>72.4</td>" in html
    # "purity" defaulted to the /fit default (linear + interactions) -- no squared terms
    purity = html[html.index('id="purity"') : html.index("<h2>Best predicted")]
    assert "temp^2" not in purity


def test_report_svg_and_tables_only(client: TestClient) -> None:
    body = {"design": _ccd_document(), "responses": [{"response": "yield"}]}
    svg = client.post("/v1/analysis/report", json={**body, "image_format": "svg"})
    assert svg.status_code == 200 and "<svg" in svg.text
    bare = client.post("/v1/analysis/report", json={**body, "figures": False})
    assert bare.status_code == 200
    assert "<svg" not in bare.text and "data:image/png" not in bare.text


def test_report_lists_fit_warnings(client: TestClient) -> None:
    design = full_factorial([ContinuousFactor("A", 0, 1), ContinuousFactor("B", 0, 1)])
    doc = design.with_response("y", [1.0, 3.0, 2.0, 7.0]).to_dict()
    response = client.post(
        "/v1/analysis/report", json={"design": doc, "responses": [{"response": "y"}]}
    )
    assert response.status_code == 200
    assert "<h2>Warnings</h2>" in response.text
    assert "y: saturated_model" in response.text


def test_report_unknown_response_is_a_validation_error(client: TestClient) -> None:
    response = client.post(
        "/v1/analysis/report",
        json={"design": _ccd_document(), "responses": [{"response": "yeild"}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_report_rejects_duplicate_and_empty_responses(client: TestClient) -> None:
    doc = _ccd_document()
    dup = client.post(
        "/v1/analysis/report",
        json={"design": doc, "responses": [{"response": "yield"}, {"response": "yield"}]},
    )
    assert dup.status_code == 422
    empty = client.post("/v1/analysis/report", json={"design": doc, "responses": []})
    assert empty.status_code == 422


def test_report_response_count_is_capped() -> None:
    client = TestClient(create_app(limits=Limits(max_goals=1)))
    response = client.post(
        "/v1/analysis/report",
        json={
            "design": _ccd_document(),
            "responses": [{"response": "yield"}, {"response": "purity"}],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "limit_exceeded"

