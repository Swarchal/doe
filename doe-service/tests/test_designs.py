"""HTTP-level tests for the Milestone 2 designs router (``docs/WEBSERVICE_BUILD.md`` §2).

Anchored the same way the library anchors known designs (``CLAUDE.md``): textbook run
counts and defining relations, through the HTTP layer rather than calling ``doe``
directly, so these tests exercise the real request/response wire contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _continuous(name: str, low: float, high: float) -> dict[str, Any]:
    return {"type": "continuous", "name": name, "low": low, "high": high}


def _mixture(name: str, low: float = 0.0, high: float = 1.0) -> dict[str, Any]:
    return {"type": "mixture", "name": name, "low": low, "high": high}


def _post(client: TestClient, path: str, body: dict[str, Any]) -> Any:
    response = client.post(path, json=body)
    return response


# --------------------------------------------------------------------------- #
# generation: textbook run counts / defining relations
# --------------------------------------------------------------------------- #


def test_full_factorial_run_count_is_2_to_the_k(client: TestClient) -> None:
    body = {
        "factors": [_continuous("a", 0, 1), _continuous("b", 0, 1), _continuous("c", 0, 1)]
    }
    response = _post(client, "/v1/designs/full-factorial", body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["design"]["runs"]) == 2**3
    assert data["warnings"] == []


def test_fractional_factorial_2_4_1_defining_relation(client: TestClient) -> None:
    """The 2^(4-1) design with generator D=ABC has defining relation I=ABCD: the coded
    product A*B*C*D is +1 for every run."""
    factors = [
        _continuous("A", -1, 1),
        _continuous("B", -1, 1),
        _continuous("C", -1, 1),
        _continuous("D", -1, 1),
    ]
    body = {"factors": factors, "generators": ["D=ABC"]}
    response = _post(client, "/v1/designs/fractional-factorial", body)
    assert response.status_code == 200, response.text
    data = response.json()
    runs = data["design"]["runs"]
    assert len(runs) == 2 ** (4 - 1)
    for run in runs:
        # natural units here equal coded units since low=-1, high=1
        product = run["A"] * run["B"] * run["C"] * run["D"]
        assert product == pytest.approx(1.0)


def test_plackett_burman_size_for_7_factors_is_8(client: TestClient) -> None:
    factors = [_continuous(f"x{i}", 0, 1) for i in range(7)]
    response = _post(client, "/v1/designs/plackett-burman", {"factors": factors})
    assert response.status_code == 200, response.text
    assert len(response.json()["design"]["runs"]) == 8


def test_central_composite_faced_run_count(client: TestClient) -> None:
    factors = [_continuous("temp", 20, 80), _continuous("time", 0, 10)]
    body = {"factors": factors, "alpha": "faced", "center": 4}
    response = _post(client, "/v1/designs/central-composite", body)
    assert response.status_code == 200, response.text
    data = response.json()
    # 2^2 factorial core + 2*2 axial + 4 center = 12
    assert len(data["design"]["runs"]) == 12
    assert data["design"]["point_types"].count("factorial") == 4
    assert data["design"]["point_types"].count("axial") == 4
    assert data["design"]["point_types"].count("center") == 4


def test_box_behnken_k3_run_count(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1), _continuous("c", 0, 1)]
    body = {"factors": factors, "center": 3}
    response = _post(client, "/v1/designs/box-behnken", body)
    assert response.status_code == 200, response.text
    # C(3, 2) * 4 edge runs + 3 center = 15
    assert len(response.json()["design"]["runs"]) == 15


def test_definitive_screening_run_count_is_2k_plus_1(client: TestClient) -> None:
    # k=4: order-4 conference matrix is directly constructible (q=3 is prime), so no
    # fake factors are auto-added and the run count is exactly 2k + 1.
    factors = [_continuous(f"x{i}", 0, 1) for i in range(4)]
    response = _post(client, "/v1/designs/definitive-screening", {"factors": factors})
    assert response.status_code == 200, response.text
    assert len(response.json()["design"]["runs"]) == 2 * 4 + 1


def test_simplex_lattice_3_2_has_six_points(client: TestClient) -> None:
    factors = [_mixture("A"), _mixture("B"), _mixture("C")]
    body = {"factors": factors, "degree": 2}
    response = _post(client, "/v1/designs/simplex-lattice", body)
    assert response.status_code == 200, response.text
    data = response.json()
    runs = data["design"]["runs"]
    assert len(runs) == 6
    for run in runs:
        assert run["A"] + run["B"] + run["C"] == pytest.approx(1.0)


def test_simplex_centroid_2_to_the_k_minus_1(client: TestClient) -> None:
    factors = [_mixture("A"), _mixture("B"), _mixture("C")]
    response = _post(client, "/v1/designs/simplex-centroid", {"factors": factors})
    assert response.status_code == 200, response.text
    assert len(response.json()["design"]["runs"]) == 2**3 - 1


def test_extreme_vertices_smoke(client: TestClient) -> None:
    factors = [_mixture("A", 0.1, 0.8), _mixture("B", 0.2, 0.9)]
    response = _post(client, "/v1/designs/extreme-vertices", {"factors": factors})
    assert response.status_code == 200, response.text
    runs = response.json()["design"]["runs"]
    assert len(runs) >= 2
    for run in runs:
        assert run["A"] + run["B"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# space-filling: seed echo, sobol power-of-two enforcement
# --------------------------------------------------------------------------- #


def test_space_filling_lhs_run_count_and_seed_echo(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {"factors": factors, "sampler": "lhs", "n_runs": 5, "seed": 7}
    response = _post(client, "/v1/designs/space-filling", body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["design"]["runs"]) == 5
    assert data["design"]["meta"]["seed"] == 7


def test_space_filling_lhs_draws_a_seed_when_omitted(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    body = {"factors": factors, "sampler": "lhs", "n_runs": 4}
    response = _post(client, "/v1/designs/space-filling", body)
    assert response.status_code == 200, response.text
    seed = response.json()["design"]["meta"]["seed"]
    assert isinstance(seed, int)


def test_space_filling_sobol_non_power_of_two_is_422_infeasible(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    body = {"factors": factors, "sampler": "sobol", "n_runs": 5}
    response = _post(client, "/v1/designs/space-filling", body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "infeasible"
    assert "power-of-two" in error["message"]
    assert "4" in error["message"] and "8" in error["message"]


def test_space_filling_sobol_power_of_two_succeeds(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    body = {"factors": factors, "sampler": "sobol", "n_runs": 8, "seed": 3}
    response = _post(client, "/v1/designs/space-filling", body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["design"]["runs"]) == 8
    assert data["design"]["meta"]["seed"] == 3


# --------------------------------------------------------------------------- #
# optimal / augment: search report + seed echo
# --------------------------------------------------------------------------- #


def test_optimal_returns_design_and_search_report_with_seed_echo(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {
        "factors": factors,
        "n_runs": 6,
        "model": "linear",
        "criterion": "D",
        "n_restarts": 3,
        "max_iter": 20,
        "seed": 42,
    }
    response = _post(client, "/v1/designs/optimal", body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["design"]["runs"]) == 6
    assert data["design"]["meta"]["seed"] == 42
    search = data["search"]
    assert search["criterion"] == "D"
    assert search["n_restarts"] == 3
    assert isinstance(search["score"], float)
    assert isinstance(search["d_efficiency"], float)
    assert isinstance(search["converged"], bool)


def test_optimal_region_shape_mismatch_is_422_infeasible(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    body = {
        "factors": factors,
        "n_runs": 6,
        "model": "linear",
        "region": [[0.0], [1.0]],  # 1 column but 2 factors
    }
    response = _post(client, "/v1/designs/optimal", body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


# --------------------------------------------------------------------------- #
# optimal / augment: server-chosen auto-parallelism (Limits.optimal_n_jobs)
# --------------------------------------------------------------------------- #


def _spy_on_coordinate_exchange(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the router's ``coordinate_exchange`` to record the ``n_jobs`` it is handed."""
    import doe_service.routers.designs as designs_router

    seen: dict[str, Any] = {}
    original = designs_router._coordinate_exchange

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["n_jobs"] = kwargs.get("n_jobs")
        return original(*args, **kwargs)

    monkeypatch.setattr(designs_router, "_coordinate_exchange", spy)
    return seen


def _optimal_body(n_runs: int) -> dict[str, Any]:
    return {
        "factors": [_continuous(n, 0, 1) for n in ("a", "b", "c", "d")],
        "n_runs": n_runs,
        "model": "quadratic",
        "criterion": "D",
        "seed": 0,
    }


def test_default_service_never_parallelises_optimal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_on_coordinate_exchange(monkeypatch)
    client = TestClient(create_app())  # DEFAULT_LIMITS -> disabled
    response = _post(client, "/v1/designs/optimal", _optimal_body(30))
    assert response.status_code == 200, response.text
    assert seen["n_jobs"] == 1


def test_configured_limits_parallelise_large_optimal(monkeypatch: pytest.MonkeyPatch) -> None:
    from doe_service.limits import Limits

    seen = _spy_on_coordinate_exchange(monkeypatch)
    limits = Limits(optimal_parallel_min_runs=24, optimal_parallel_max_workers=2)
    client = TestClient(create_app(limits=limits))

    small = _post(client, "/v1/designs/optimal", _optimal_body(16))
    assert small.status_code == 200, small.text
    assert seen["n_jobs"] == 1  # below threshold (24) stays single-process

    large = _post(client, "/v1/designs/optimal", _optimal_body(30))
    assert large.status_code == 200, large.text
    assert seen["n_jobs"] == 2  # at/above threshold uses the configured workers
    assert len(large.json()["design"]["runs"]) == 30  # and still a valid design


def test_augment_holds_existing_rows_fixed_and_tags_point_types(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    base = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    assert base.status_code == 200, base.text
    base_design = base.json()["design"]
    n_existing = len(base_design["runs"])

    body = {
        "design": base_design,
        "n_runs": 2,
        "model": "linear",
        "n_restarts": 3,
        "max_iter": 20,
        "seed": 11,
    }
    response = _post(client, "/v1/designs/augment", body)
    assert response.status_code == 200, response.text
    data = response.json()
    point_types = data["design"]["point_types"]
    assert point_types.count("existing") == n_existing
    assert point_types.count("augment") == 2
    assert len(data["design"]["runs"]) == n_existing + 2
    search = data["search"]
    assert search["criterion"] == "D"
    assert search["n_restarts"] == 3


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #


def test_candidates_grid_for_box_factors(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    response = _post(client, "/v1/designs/candidates", {"factors": factors, "levels": 3})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["kind"] == "grid"
    assert len(data["points"]) == 3**2


def test_candidates_mixture_for_mixture_factors(client: TestClient) -> None:
    factors = [_mixture("A"), _mixture("B"), _mixture("C")]
    response = _post(client, "/v1/designs/candidates", {"factors": factors, "resolution": 4})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["kind"] == "mixture"
    assert len(data["points"]) > 0
    for point in data["points"]:
        assert sum(point) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# validate: 200 with {valid, errors} both ways
# --------------------------------------------------------------------------- #


def test_validate_valid_document_returns_200_valid_true(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    response = _post(client, "/v1/designs/validate", {"design": document})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data == {"valid": True, "errors": []}


def test_validate_invalid_document_returns_200_valid_false(client: TestClient) -> None:
    document = {
        "schema_version": "1.0",
        "factors": [_continuous("a", 0, 1)],
        "runs": [{}],  # missing value for factor 'a'
    }
    response = _post(client, "/v1/designs/validate", {"design": document})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["valid"] is False
    assert any("a" in e for e in data["errors"])


# --------------------------------------------------------------------------- #
# operations: randomize / replicate / project / responses
# --------------------------------------------------------------------------- #


def test_randomize_shuffles_and_echoes_seed(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    response = _post(client, "/v1/designs/randomize", {"design": document, "seed": 5})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["design"]["meta"]["random_seed"] == 5
    assert len(data["design"]["runs"]) == len(document["runs"])


def test_randomize_draws_a_seed_when_omitted(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    response = _post(client, "/v1/designs/randomize", {"design": document})
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["design"]["meta"]["random_seed"], int)


def test_replicate_doubles_run_count(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    response = _post(client, "/v1/designs/replicate", {"design": document, "n": 2})
    assert response.status_code == 200, response.text
    assert len(response.json()["design"]["runs"]) == len(document["runs"]) * 2


def test_project_narrows_factor_set(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1), _continuous("c", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    response = _post(client, "/v1/designs/project", {"design": document, "factors": ["a", "b"]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [f["name"] for f in data["design"]["factors"]] == ["a", "b"]
    for run in data["design"]["runs"]:
        assert "c" not in run


def test_responses_attaches_aligned_columns(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]
    n = len(document["runs"])

    body = {"design": document, "responses": {"yield": [float(i) for i in range(n)]}}
    response = _post(client, "/v1/designs/responses", body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert [run["yield"] for run in data["design"]["runs"]] == [float(i) for i in range(n)]


def test_responses_length_mismatch_is_422_infeasible(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1)]
    generated = _post(client, "/v1/designs/full-factorial", {"factors": factors})
    document = generated.json()["design"]

    body = {"design": document, "responses": {"yield": [1.0]}}  # wrong length
    response = _post(client, "/v1/designs/responses", body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


# --------------------------------------------------------------------------- #
# malformed design document -> 422 validation_error listing all problems
# --------------------------------------------------------------------------- #


def test_malformed_design_document_lists_all_problems(client: TestClient) -> None:
    document = {
        "schema_version": "1.0",
        "factors": [_continuous("temp", 20.0, 80.0)],
        "runs": [{}, {"temp": 50.0}],  # run 0 missing 'temp'
        "point_types": ["factorial"],  # 1 entry for 2 runs
    }
    response = _post(client, "/v1/designs/randomize", {"design": document})
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert len(error["errors"]) >= 2
    assert any("temp" in e and "missing" in e for e in error["errors"])
    assert any("point_types" in e for e in error["errors"])


def test_full_factorial_bad_factor_bounds_is_422_infeasible(client: TestClient) -> None:
    # high <= low is rejected by ContinuousFactor's own constructor, still routed
    # through call_library since factor construction happens inside the wrapped call.
    body = {"factors": [_continuous("a", 10, 5)]}
    response = _post(client, "/v1/designs/full-factorial", body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "infeasible"


def test_no_response_body_contains_nan_literal(client: TestClient) -> None:
    factors = [_continuous("a", 0, 1), _continuous("b", 0, 1)]
    response = _post(client, "/v1/designs/central-composite", {"factors": factors})
    assert response.status_code == 200, response.text
    assert "NaN" not in response.text
