"""The full v1 route table from docs/WEBSERVICE_API.md is mounted.

This test outlives the stubs: it pins the wire-level route set through every
implementation milestone (docs/WEBSERVICE_BUILD.md) — a route rename or removal is a
breaking API change and must fail here first.
"""

from fastapi.testclient import TestClient

from doe_service.main import create_app

EXPECTED_V1_ROUTES = {
    ("GET", "/v1/health"),
    # design generation
    ("POST", "/v1/designs/full-factorial"),
    ("POST", "/v1/designs/fractional-factorial"),
    ("POST", "/v1/designs/plackett-burman"),
    ("POST", "/v1/designs/definitive-screening"),
    ("POST", "/v1/designs/central-composite"),
    ("POST", "/v1/designs/box-behnken"),
    ("POST", "/v1/designs/space-filling"),
    ("POST", "/v1/designs/simplex-lattice"),
    ("POST", "/v1/designs/simplex-centroid"),
    ("POST", "/v1/designs/extreme-vertices"),
    ("POST", "/v1/designs/optimal"),
    ("POST", "/v1/designs/augment"),
    ("POST", "/v1/designs/candidates"),
    # design operations
    ("POST", "/v1/designs/validate"),
    ("POST", "/v1/designs/randomize"),
    ("POST", "/v1/designs/replicate"),
    ("POST", "/v1/designs/project"),
    ("POST", "/v1/designs/responses"),
    # analysis
    ("POST", "/v1/analysis/fit"),
    ("POST", "/v1/analysis/anova"),
    ("POST", "/v1/analysis/predict"),
    ("POST", "/v1/analysis/diagnostics"),
    ("POST", "/v1/analysis/coverage"),
    # optimization
    ("POST", "/v1/optimize/stationary-point"),
    ("POST", "/v1/optimize/optimum"),
    ("POST", "/v1/optimize/desirability"),
    # plot data
    ("POST", "/v1/plot-data/surface"),
    ("POST", "/v1/plot-data/interactions"),
    ("POST", "/v1/plot-data/ternary"),
    ("POST", "/v1/plot-data/alias"),
}

#: Routes still behind the ``not_implemented()`` 501 stub. It was narrowed deliberately,
#: milestone by milestone (rather than deleted), as each router landed real typed request
#: models and a bare ``POST {}`` began returning 422 instead of 501.
#: Every v1 route is now implemented (Milestones 1-5 all landed), so nothing is stubbed.
#: The set stays here, empty, as the anchor for `test_unimplemented_routes_return_501`:
#: if a future route is added as a 501 stub, add it here and the test starts pinning it
#: again.
STILL_STUBBED_ROUTES: set[tuple[str, str]] = set()


def test_v1_route_table_is_mounted() -> None:
    # Enumerate via the OpenAPI schema — the public surface — rather than
    # introspecting router internals.
    spec = create_app().openapi()
    mounted = {
        (method.upper(), path)
        for path, operations in spec["paths"].items()
        for method in operations
    }
    assert EXPECTED_V1_ROUTES <= mounted


def test_unimplemented_routes_return_501() -> None:
    client = TestClient(create_app())
    for _, path in sorted(STILL_STUBBED_ROUTES):
        assert client.post(path).status_code == 501, path


def test_designs_routes_are_no_longer_stubs() -> None:
    # The Milestone 2 designs router now has typed request models, so a bare ``POST {}``
    # is a 422 (missing required fields) rather than the stub's 501 -- this is the
    # milestone's positive counterpart to the narrowed 501 test above.
    client = TestClient(create_app())
    designs_routes = {
        path for method, path in EXPECTED_V1_ROUTES if method == "POST" and "/v1/designs/" in path
    }
    for path in sorted(designs_routes):
        assert client.post(path).status_code != 501, path


def test_analysis_routes_are_no_longer_stubs() -> None:
    # The Milestone 3 analysis router now has typed request models, so a bare ``POST {}``
    # is a 422 (missing required fields) rather than the stub's 501 -- this is the
    # milestone's positive counterpart to the narrowed 501 test above.
    client = TestClient(create_app())
    analysis_routes = {
        path
        for method, path in EXPECTED_V1_ROUTES
        if method == "POST" and "/v1/analysis/" in path
    }
    for path in sorted(analysis_routes):
        assert client.post(path, json={}).status_code == 422, path


def test_optimize_routes_are_no_longer_stubs() -> None:
    # The Milestone 4 optimize router now has typed request models, so a bare
    # ``POST {}`` is a 422 (missing required fields) rather than the stub's 501 -- this
    # is the milestone's positive counterpart to the narrowed 501 test above.
    client = TestClient(create_app())
    optimize_routes = {
        path
        for method, path in EXPECTED_V1_ROUTES
        if method == "POST" and "/v1/optimize/" in path
    }
    for path in sorted(optimize_routes):
        assert client.post(path, json={}).status_code == 422, path


def test_plot_data_routes_are_no_longer_stubs() -> None:
    # The Milestone 5 plot-data router now has typed request models, so a bare
    # ``POST {}`` is a 422 (missing required fields) rather than the stub's 501 -- the
    # last milestone's positive counterpart to the (now empty) 501 test above.
    client = TestClient(create_app())
    plot_routes = {
        path
        for method, path in EXPECTED_V1_ROUTES
        if method == "POST" and "/v1/plot-data/" in path
    }
    for path in sorted(plot_routes):
        assert client.post(path, json={}).status_code == 422, path
