"""HTTP-level tests for the Phase-5 endpoints: split-plot / blocking generators, the
categorical DSD passthrough, and the split-plot ``fit-gls`` analysis route.

Anchored like the rest of the suite (``CLAUDE.md``): textbook run counts / structure
through the real request/response wire contract, plus the two facts the wire additions
exist to carry -- ``hard_to_change`` on factors and ``whole_plots`` on a design survive
the round-trip only when set, so the existing generation contracts are undisturbed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _continuous(name: str, low: float, high: float, *, hard: bool = False) -> dict[str, Any]:
    f: dict[str, Any] = {"type": "continuous", "name": name, "low": low, "high": high}
    if hard:
        f["hard_to_change"] = True
    return f


# --------------------------------------------------------------------------- #
# split-plot generation
# --------------------------------------------------------------------------- #


def test_split_plot_sets_whole_plots_and_carries_hard_to_change(client: TestClient) -> None:
    body = {
        "factors": [
            _continuous("temp", 100, 200, hard=True),
            _continuous("conc", 1, 5),
        ],
        "seed": 1,
    }
    response = client.post("/v1/designs/split-plot", json=body)
    assert response.status_code == 200, response.text
    design = response.json()["design"]

    # whole_plots is present (this is a split-plot design) and one id per run.
    assert "whole_plots" in design
    assert len(design["whole_plots"]) == len(design["runs"])
    # two whole plots (temp at -1 and +1), two runs each = 4 runs.
    assert sorted(set(design["whole_plots"])) == [0, 1]

    # hard_to_change survives on the whole-plot factor only.
    by_name = {f["name"]: f for f in design["factors"]}
    assert by_name["temp"].get("hard_to_change") is True
    assert "hard_to_change" not in by_name["conc"]


def test_split_plot_requires_a_hard_to_change_factor(client: TestClient) -> None:
    body = {"factors": [_continuous("a", 0, 1), _continuous("b", 0, 1)]}
    response = client.post("/v1/designs/split-plot", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible"


# --------------------------------------------------------------------------- #
# blocking generation
# --------------------------------------------------------------------------- #


def test_randomized_complete_block_int_treatments(client: TestClient) -> None:
    body = {"n_treatments": 4, "n_blocks": 3, "seed": 7}
    response = client.post("/v1/designs/randomized-complete-block", json=body)
    assert response.status_code == 200, response.text
    design = response.json()["design"]
    # every treatment once per block => 4 * 3 runs; block is a reserved categorical.
    assert len(design["runs"]) == 12
    assert any(f["name"] == "block" for f in design["factors"])


def test_randomized_complete_block_needs_exactly_one_treatment_spec(
    client: TestClient,
) -> None:
    body = {"factors": [_continuous("a", 0, 1)], "n_treatments": 3, "n_blocks": 2}
    response = client.post("/v1/designs/randomized-complete-block", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible"


def test_latin_square_run_count_is_t_squared(client: TestClient) -> None:
    response = client.post("/v1/designs/latin-square", json={"treatments": 4, "seed": 3})
    assert response.status_code == 200, response.text
    design = response.json()["design"]
    assert len(design["runs"]) == 4**2


def test_blocked_factorial_confounds_into_blocks(client: TestClient) -> None:
    factors = [_continuous(n, -1, 1) for n in ("A", "B", "C")]
    body = {"factors": factors, "block_generators": ["ABC"], "seed": 5}
    response = client.post("/v1/designs/blocked-factorial", json=body)
    assert response.status_code == 200, response.text
    design = response.json()["design"]
    assert len(design["runs"]) == 2**3
    assert any(f["name"] == "block" for f in design["factors"])
    assert "confounded_with_blocks" in design["meta"]


# --------------------------------------------------------------------------- #
# categorical DSD passthrough (no new endpoint -- the existing route accepts it)
# --------------------------------------------------------------------------- #


def test_definitive_screening_accepts_two_level_categorical(client: TestClient) -> None:
    body = {
        "factors": [
            _continuous("x1", -1, 1),
            _continuous("x2", -1, 1),
            {"type": "categorical", "name": "cat", "levels": ["lo", "hi"]},
        ]
    }
    response = client.post("/v1/designs/definitive-screening", json=body)
    assert response.status_code == 200, response.text
    design = response.json()["design"]
    # The Jones-Nachtsheim (2013) categorical-augment path ran (the exact run count is
    # pinned in the library tests); here it is enough that it produced a design whose
    # categorical factor takes both of its declared levels.
    assert len(design["runs"]) >= 2 * len(body["factors"])
    assert {r["cat"] for r in design["runs"]} == {"lo", "hi"}


# --------------------------------------------------------------------------- #
# fit-gls
# --------------------------------------------------------------------------- #


def _split_plot_with_response(client: TestClient) -> dict[str, Any]:
    """A 16-run split-plot design carrying a synthetic ``y`` column."""
    body = {
        "factors": [
            _continuous("temp", 100, 200, hard=True),
            _continuous("conc", 1, 5),
            _continuous("ph", 6, 8),
        ],
        "n_whole_plot_reps": 2,
        "seed": 1,
    }
    design = client.post("/v1/designs/split-plot", json=body).json()["design"]
    n = len(design["runs"])
    # deterministic pseudo-response so the fit is well posed but not degenerate.
    values = [50.0 + (i % 5) - 2.0 * ((i // 4) % 2) for i in range(n)]
    with_resp = client.post(
        "/v1/designs/responses", json={"design": design, "responses": {"y": values}}
    )
    assert with_resp.status_code == 200, with_resp.text
    return with_resp.json()["design"]


def test_fit_gls_returns_variance_components(client: TestClient) -> None:
    design = _split_plot_with_response(client)
    response = client.post(
        "/v1/analysis/fit-gls", json={"design": design, "response": "y", "model": "linear"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    # base fit shape plus the split-plot extras.
    assert {"terms", "r_squared", "sigma2_wp", "n_whole_plots", "dof_terms"} <= data.keys()
    assert data["n_whole_plots"] == 4
    assert data["sigma2_wp"] is not None
    assert set(data["dof_terms"]) == {t["term"] for t in data["terms"]}


def test_fit_gls_rejects_a_design_without_whole_plots(client: TestClient) -> None:
    """A plain (non-split-plot) design has no whole-plot structure -> 422 infeasible."""
    ff = client.post(
        "/v1/designs/full-factorial",
        json={"factors": [_continuous("a", -1, 1), _continuous("b", -1, 1)]},
    ).json()["design"]
    with_resp = client.post(
        "/v1/designs/responses",
        json={"design": ff, "responses": {"y": [1.0, 2.0, 3.0, 4.0]}},
    ).json()["design"]
    response = client.post(
        "/v1/analysis/fit-gls", json={"design": with_resp, "response": "y"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible"


# --------------------------------------------------------------------------- #
# wire round-trip: the additions are invisible unless set
# --------------------------------------------------------------------------- #


def test_plain_design_document_gains_no_new_keys(client: TestClient) -> None:
    """A non-split-plot design must serialize with no ``whole_plots`` key and no
    ``hard_to_change`` on its factors -- the invariant the existing contracts rely on."""
    design = client.post(
        "/v1/designs/full-factorial",
        json={"factors": [_continuous("a", -1, 1), _continuous("b", -1, 1)]},
    ).json()["design"]
    assert "whole_plots" not in design
    for factor in design["factors"]:
        assert "hard_to_change" not in factor
