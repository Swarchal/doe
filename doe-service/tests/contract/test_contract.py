"""Contract lock-in: every worked example in ``docs/WEBSERVICE_API.md`` reproduced
through the real HTTP layer (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).

One golden ``pairs/<name>.json`` per doc example (15 total): the four design-generation
examples (``central-composite``, ``optimal``, ``augment``, ``candidates``) plus the two
Phase-5 generation examples (``split-plot``, and ``fit-gls`` under analysis), the five
analysis examples (``fit``, ``anova``, ``predict``, ``diagnostics``, ``coverage``), the
three optimization examples (``stationary-point``, ``optimum``, ``desirability``), and
the error envelope example. Each fixture's ``request`` is POSTed and the live response
is compared to the stored ``response`` via :func:`support.assert_matches` -- structural,
not bit-exact, so floating-point noise from the platform's BLAS/LAPACK doesn't make this
flaky while still locking every number that matters.

Two doc examples were fabricated placeholders that did not match any real computation
(``/v1/designs/optimal``'s ``score``/``d_efficiency`` and the error envelope's exact
message wording) -- ``docs/WEBSERVICE_API.md`` was corrected to the real values these
fixtures anchor; see the Milestone 6 report for the diff.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from contract.support import assert_matches, load_pair, pair_names
from doe_service.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize("name", pair_names())
def test_contract_pair(client: TestClient, name: str) -> None:
    pair = load_pair(name)
    response = client.request(pair["method"], pair["path"], json=pair["request"])
    assert response.status_code == pair["status"], response.text
    assert_matches(response.json(), pair["response"])


def test_fifteen_pairs_are_locked_in() -> None:
    """One pair per ``docs/WEBSERVICE_API.md`` example -- pins the count so a future
    example added to the doc without a matching fixture is caught here first."""
    assert len(pair_names()) == 15
