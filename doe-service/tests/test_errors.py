"""Tests for the error envelope and exception → status mapping (Milestone 1).

Mounts a small throwaway app rather than the real routers (which are still 501 stubs
in this milestone) so the handler behaviour can be exercised end to end.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from doe_service.convert import call_library, design_from_document
from doe_service.errors import register_exception_handlers
from doe_service.limits import LimitExceeded
from doe_service.schemas.design import DesignDocument


class _Body(BaseModel):
    name: str


def _make_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/test/design")
    def _design(document: DesignDocument) -> dict[str, Any]:
        design_from_document(document.model_dump())
        return {"ok": True}

    @app.post("/test/required-field")
    def _required_field(body: _Body) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/test/limit")
    def _limit() -> dict[str, Any]:
        raise LimitExceeded("runs (20000) exceeds the cap of 10000")

    @app.post("/test/infeasible")
    def _infeasible() -> dict[str, Any]:
        def _raises() -> None:
            raise ValueError("sobol run count must be a power of two; nearest valid: 8, 16")

        call_library(_raises)
        return {"ok": True}

    @app.post("/test/boom")
    def _boom() -> dict[str, Any]:
        raise RuntimeError("something truly unexpected")

    @app.post("/test/not-implemented")
    def _stub() -> dict[str, Any]:
        from doe_service.errors import not_implemented

        raise not_implemented()

    return app


_INVALID_DESIGN_DOCUMENT: dict[str, Any] = {
    "schema_version": "1.0",
    "name": "bad",
    "factors": [{"type": "continuous", "name": "temp", "low": 20.0, "high": 80.0}],
    "runs": [{}, {"temp": 50.0}],  # run 0 missing 'temp'
    "point_types": ["factorial"],  # 1 entry for 2 runs
    "meta": {},
}


def test_malformed_design_returns_envelope_with_all_validator_errors() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/design", json=_INVALID_DESIGN_DOCUMENT)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert len(body["error"]["errors"]) >= 2
    assert any("temp" in e for e in body["error"]["errors"])
    assert any("point_types" in e for e in body["error"]["errors"])


def test_error_envelope_shape_matches_the_spec() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/design", json=_INVALID_DESIGN_DOCUMENT)

    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "errors"}
    assert isinstance(body["error"]["code"], str)
    assert isinstance(body["error"]["message"], str)
    assert isinstance(body["error"]["errors"], list)


def test_pydantic_shape_failure_returns_422_validation_error() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/required-field", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert len(body["error"]["errors"]) >= 1


def test_malformed_json_body_returns_400_malformed() -> None:
    client = TestClient(_make_app())

    response = client.post(
        "/test/required-field",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "malformed"


def test_limit_exceeded_returns_422() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/limit")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "limit_exceeded"
    assert "10000" in body["error"]["message"]


def test_infeasible_value_error_returns_422() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/infeasible")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "infeasible"
    assert "sobol" in body["error"]["message"]


def test_unexpected_exception_returns_500_internal_without_leaking_detail() -> None:
    client = TestClient(_make_app(), raise_server_exceptions=False)

    response = client.post("/test/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal"
    assert "truly unexpected" not in body["error"]["message"]


def test_not_implemented_still_returns_501_alongside_registered_handlers() -> None:
    client = TestClient(_make_app())

    response = client.post("/test/not-implemented")

    assert response.status_code == 501


def test_infeasible_is_not_wrapped_as_500() -> None:
    """Sanity check that Infeasible is caught by its own handler, not the catch-all."""
    client = TestClient(_make_app())

    response = client.post("/test/infeasible")

    assert response.status_code != 500
