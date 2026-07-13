"""OpenAPI polish checks (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6 third bullet).

``test_routes.py`` already pins the mounted route *set*; this module checks the schema
itself builds cleanly, that every operation has a summary/description (FastAPI derives
these from each route's docstring) and a non-``dict[str, Any]`` request/response body,
and that the Milestone 6 examples (``schemas/results.py``, ``schemas/design.py``,
``schemas/generation.py``) are actually wired into the generated schema.
"""

from __future__ import annotations

from typing import Any

from doe_service.main import create_app


def _operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method.upper(), path, operation)
        for path, methods in spec["paths"].items()
        for method, operation in methods.items()
    ]


def test_openapi_schema_builds_without_error() -> None:
    spec = create_app().openapi()
    assert spec["openapi"]
    assert spec["paths"]


def test_openapi_documents_every_v1_route() -> None:
    # 35 POST compute routes (design generation/operations/candidates including the
    # Phase-5 split-plot/blocking generators, analysis including split-plot fit-gls,
    # optimize, plot-data) plus GET /v1/health.
    spec = create_app().openapi()
    operations = _operations(spec)
    post_routes = [(m, p) for m, p, _ in operations if m == "POST" and p.startswith("/v1/")]
    get_routes = [(m, p) for m, p, _ in operations if m == "GET" and p == "/v1/health"]
    assert len(post_routes) == 35
    assert len(get_routes) == 1


def test_every_v1_operation_has_a_docstring_derived_description() -> None:
    spec = create_app().openapi()
    for method, path, operation in _operations(spec):
        if not path.startswith("/v1/") or path == "/v1/health":
            continue
        assert operation.get("description"), f"{method} {path} has no docstring-derived description"


def test_every_v1_post_route_has_a_typed_request_body_schema() -> None:
    """No endpoint accepts a bare/untyped JSON body -- every POST operation's request
    body resolves to a named component schema (a Pydantic model), not an inline
    ``additionalProperties`` free-for-all."""
    spec = create_app().openapi()
    for method, path, operation in _operations(spec):
        if method != "POST" or not path.startswith("/v1/"):
            continue
        body = operation["requestBody"]["content"]["application/json"]["schema"]
        assert "$ref" in body, f"{method} {path} request body is not a named schema: {body}"


def test_every_v1_post_route_has_a_typed_200_response_schema() -> None:
    spec = create_app().openapi()
    for method, path, operation in _operations(spec):
        if method != "POST" or not path.startswith("/v1/"):
            continue
        ok = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in ok, f"{method} {path} 200 response is not a named schema: {ok}"


def test_milestone_6_examples_are_wired_into_the_schema() -> None:
    spec = create_app().openapi()
    schemas = spec["components"]["schemas"]
    for name in (
        "FitResponse",
        "AnovaResponse",
        "PredictResponse",
        "DiagnosticsResponse",
        "CoverageResponse",
        "StationaryPointResponse",
        "OptimumResponse",
        "DesirabilityResponse",
        "CandidatesResponse",
        "CandidatesRequest",
        "CentralCompositeRequest",
    ):
        assert schemas[name].get("examples"), f"{name} has no OpenAPI example"
