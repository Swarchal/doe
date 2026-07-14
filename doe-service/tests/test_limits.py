"""HTTP-level tests for Milestone 6 limit enforcement (``docs/WEBSERVICE_BUILD.md`` §6).

Router unit tests already cover ``max_goals`` (``test_optimize.py``) and
``max_resolution`` (``test_plot_data.py``); this module covers the gaps closed in
Milestone 6: ``max_factors``, ``max_runs``, the ``n_restarts``/``max_iter`` search
budget, ``max_region_rows``, and the ``max_body_bytes`` Content-Length middleware.
Every ``limit_exceeded`` message is checked for naming both the cap and the ceiling
(``docs/WEBSERVICE_API.md`` "Errors": "message names the cap and the ceiling").
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from doe_service.limits import DEFAULT_LIMITS, Limits
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _continuous(name: str, low: float, high: float) -> dict[str, Any]:
    return {"type": "continuous", "name": name, "low": low, "high": high}


# --------------------------------------------------------------------------- #
# max_factors
# --------------------------------------------------------------------------- #


def test_too_many_factors_is_422_limit_exceeded(client: TestClient) -> None:
    # levels defaults to 2, so an unchecked request would try to build 2**33 runs --
    # this only returns promptly because the factor-count cap is checked *before* the
    # generator ever runs (``designs._factors``, Milestone 6).
    factors = [_continuous(f"x{i}", 0, 1) for i in range(DEFAULT_LIMITS.max_factors + 1)]
    response = client.post("/v1/designs/full-factorial", json={"factors": factors})
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_factors) in error["message"]
    assert str(len(factors)) in error["message"]


def test_factor_count_within_cap_succeeds(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    response = client.post("/v1/designs/full-factorial", json={"factors": factors})
    assert response.status_code == 200, response.text


def test_too_many_factors_on_a_posted_design_document_is_422_limit_exceeded(
    client: TestClient,
) -> None:
    # design_from_document (convert.py) enforces the same cap for every endpoint that
    # takes a full design document rather than a bare factors list.
    factors = [_continuous(f"x{i}", 0, 1) for i in range(DEFAULT_LIMITS.max_factors + 1)]
    document = {
        "schema_version": "1.0",
        "factors": factors,
        "runs": [{f["name"]: 0.5 for f in factors}],
    }
    response = client.post("/v1/designs/randomize", json={"design": document})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


# --------------------------------------------------------------------------- #
# max_runs
# --------------------------------------------------------------------------- #


def test_too_many_runs_from_replicate_is_422_limit_exceeded(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    generated = client.post("/v1/designs/full-factorial", json={"factors": factors})
    document = generated.json()["design"]  # 4 runs
    n = DEFAULT_LIMITS.max_runs  # 4 * max_runs runs, comfortably over the cap

    response = client.post("/v1/designs/replicate", json={"design": document, "n": n})
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_runs) in error["message"]


def test_run_count_within_cap_succeeds(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = client.post("/v1/designs/full-factorial", json={"factors": factors})
    document = generated.json()["design"]

    response = client.post("/v1/designs/replicate", json={"design": document, "n": 3})
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# n_restarts x max_iter search budget (/optimal, /augment)
# --------------------------------------------------------------------------- #


def test_optimal_too_many_restarts_is_422_limit_exceeded_before_search_runs(
    client: TestClient,
) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {
        "factors": factors,
        "n_runs": 6,
        "n_restarts": DEFAULT_LIMITS.max_restarts + 1,
    }
    response = client.post("/v1/designs/optimal", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_restarts) in error["message"]


def test_optimal_too_many_iterations_is_422_limit_exceeded(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {
        "factors": factors,
        "n_runs": 6,
        "max_iter": DEFAULT_LIMITS.max_iter + 1,
    }
    response = client.post("/v1/designs/optimal", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_iter) in error["message"]


def test_optimal_n_runs_over_cap_is_422_limit_exceeded_before_search_runs(
    client: TestClient,
) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {"factors": factors, "n_runs": DEFAULT_LIMITS.max_runs + 1, "n_restarts": 1}
    response = client.post("/v1/designs/optimal", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_augment_too_many_restarts_is_422_limit_exceeded(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    base = client.post("/v1/designs/full-factorial", json={"factors": factors})
    document = base.json()["design"]

    body = {
        "design": document,
        "n_runs": 2,
        "n_restarts": DEFAULT_LIMITS.max_restarts + 1,
    }
    response = client.post("/v1/designs/augment", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_restarts) in error["message"]


def test_optimal_search_budget_within_cap_succeeds(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {
        "factors": factors,
        "n_runs": 6,
        "model": "linear",
        "n_restarts": 2,
        "max_iter": 10,
    }
    response = client.post("/v1/designs/optimal", json=body)
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# max_region_rows (/optimal, /augment, /analysis/diagnostics)
# --------------------------------------------------------------------------- #


def test_optimal_region_row_count_over_cap_is_422_limit_exceeded(client: TestClient) -> None:
    # Router-level checks (this one included) all read the module-level DEFAULT_LIMITS,
    # matching the already-landed max_goals/max_resolution pattern -- a per-app ``limits``
    # override only reaches the body-size middleware below, which is why this exercises
    # the real default cap rather than a ``create_app(limits=...)`` override. A
    # single-factor region keeps the (still ~100_001-row) JSON payload small.
    factors = [_continuous("a", 0, 1)]
    n_rows = DEFAULT_LIMITS.max_region_rows + 1
    body = {"factors": factors, "n_runs": 2, "region": [[0.0]] * n_rows}
    response = client.post("/v1/designs/optimal", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert str(DEFAULT_LIMITS.max_region_rows) in error["message"]


def test_diagnostics_region_row_count_over_cap_is_422_limit_exceeded(client: TestClient) -> None:
    from doe import ContinuousFactor, FactorSet, full_factorial

    design = full_factorial(FactorSet([ContinuousFactor("a", 0, 1)]))
    n_rows = DEFAULT_LIMITS.max_region_rows + 1
    body = {"design": design.to_dict(), "region": [[0.0]] * n_rows}
    response = client.post("/v1/analysis/diagnostics", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


# --------------------------------------------------------------------------- #
# max_body_bytes (Content-Length middleware -> 413, not 422)
# --------------------------------------------------------------------------- #


def test_oversized_body_is_413_with_the_standard_envelope() -> None:
    app = create_app(limits=Limits(max_body_bytes=100))
    client = TestClient(app)
    factors = [_continuous(f"x{i}", 0, 1) for i in range(20)]  # well over 100 bytes of JSON
    body = json.dumps({"factors": factors})
    assert len(body.encode()) > 100

    response = client.post(
        "/v1/designs/full-factorial",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert "100" in error["message"]


def test_body_within_cap_succeeds() -> None:
    app = create_app(limits=Limits(max_body_bytes=100))
    client = TestClient(app)
    body = json.dumps({"factors": [_continuous("a", 0, 1)]})
    assert len(body.encode()) <= 100

    response = client.post(
        "/v1/designs/full-factorial",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200, response.text


def test_default_body_cap_does_not_reject_ordinary_requests(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    response = client.post("/v1/designs/full-factorial", json={"factors": factors})
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# projected run counts (the cap must fire *before* the allocation it exists to prevent)
# --------------------------------------------------------------------------- #


def test_full_factorial_levels_are_capped_before_generating(client: TestClient) -> None:
    # 8 factors x 10 levels = 10**8 runs. Both inputs are within their own declared limits
    # (max_factors is 32, and `levels` has no cap), so nothing rejected this: the service
    # generated the design first and only then checked max_runs -- exhausting memory long
    # before the check. The run count must be projected from the request instead.
    body = {"factors": [_continuous(f"x{i}", 0, 1) for i in range(8)], "levels": 10}
    response = client.post("/v1/designs/full-factorial", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "limit_exceeded"
    assert "100000000" in error["message"]
    assert str(DEFAULT_LIMITS.max_runs) in error["message"]


def test_full_factorial_two_level_factor_count_is_capped_before_generating(
    client: TestClient,
) -> None:
    # 25 two-level factors = 2**25 runs, with both `factors` and `levels` inside their limits
    body = {"factors": [_continuous(f"x{i}", 0, 1) for i in range(25)], "levels": 2}
    response = client.post("/v1/designs/full-factorial", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_latin_square_treatments_are_capped_before_generating(client: TestClient) -> None:
    response = client.post("/v1/designs/latin-square", json={"treatments": 100_000})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_box_behnken_center_is_capped_before_generating(client: TestClient) -> None:
    body = {"factors": [_continuous(f"x{i}", 0, 1) for i in range(3)], "center": 10**8}
    response = client.post("/v1/designs/box-behnken", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_candidate_grid_levels_are_capped_before_generating(client: TestClient) -> None:
    body = {"factors": [_continuous(f"x{i}", 0, 1) for i in range(8)], "levels": 10}
    response = client.post("/v1/designs/candidates", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_blocked_factorial_factor_count_is_capped_before_generating(client: TestClient) -> None:
    body = {
        "factors": [_continuous(f"x{i}", 0, 1) for i in range(25)],
        "block_generators": ["ABC"],
    }
    response = client.post("/v1/designs/blocked-factorial", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_designs_within_the_projected_cap_still_generate(client: TestClient) -> None:
    # the projection must not reject legitimate requests
    body = {"factors": [_continuous(f"x{i}", 0, 1) for i in range(3)], "levels": 3}
    assert client.post("/v1/designs/full-factorial", json=body).status_code == 200
    assert client.post("/v1/designs/latin-square", json={"treatments": 4}).status_code == 200


# --------------------------------------------------------------------------- #
# max_body_bytes: the cap counts bytes, it does not merely trust Content-Length
# --------------------------------------------------------------------------- #


def test_oversized_chunked_body_without_content_length_is_still_413() -> None:
    # A chunked request declares no Content-Length, so a header-only check waves it through
    # and nothing further down the stack bounds what is read into memory.
    app = create_app(limits=Limits(max_body_bytes=1000))

    def chunks() -> Any:
        for _ in range(50):
            yield b"x" * 500  # 25 kB in total, no Content-Length

    with TestClient(app) as client:
        response = client.post(
            "/v1/designs/full-factorial",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["code"] == "limit_exceeded"


def test_body_within_the_cap_is_served_normally() -> None:
    app = create_app(limits=Limits(max_body_bytes=100_000))
    body = {"factors": [_continuous("a", 0, 1)], "levels": 2}
    with TestClient(app) as client:
        assert client.post("/v1/designs/full-factorial", json=body).status_code == 200
