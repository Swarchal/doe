"""assert_matches tolerance scoping (code review 2026-07-16, Finding 5).

The split-plot REML fit derives σ²_wp (and the GLS standard errors / t / p / CI that
depend on it) from a *flat* likelihood search whose argmin the platform's BLAS/LAPACK last
ULP fixes, so those values vary ~1e-5 between platforms. They get the loose tolerance;
every deterministic value -- including fit_gls's own coefficients -- stays locked at the
tight tolerance, so a real regression in a deterministic number is still caught.
"""

from __future__ import annotations

import pytest

from contract.support import assert_matches, loose_keys_for


def _fit_gls_like(sigma2_wp: float, coefficient: float) -> dict:
    return {"sigma2_wp": sigma2_wp, "terms": [{"term": "temp", "coefficient": coefficient}]}


def test_search_derived_field_tolerates_cross_platform_noise() -> None:
    loose = loose_keys_for("fit_gls")
    expected = _fit_gls_like(0.25, -0.625)
    # sigma2_wp off by ~4e-6 relative: within the search tolerance, but over the tight one
    actual = _fit_gls_like(0.25 * (1 + 4e-6), -0.625)
    assert_matches(actual, expected, loose_keys=loose)  # passes under the loose key


def test_search_derived_field_would_fail_without_the_loose_scope() -> None:
    # same drift, but no loose keys -> the tight default catches it (proves the scope matters)
    expected = _fit_gls_like(0.25, -0.625)
    actual = _fit_gls_like(0.25 * (1 + 4e-6), -0.625)
    with pytest.raises(AssertionError):
        assert_matches(actual, expected)


def test_deterministic_coefficient_stays_tight_even_within_fit_gls() -> None:
    loose = loose_keys_for("fit_gls")
    expected = _fit_gls_like(0.25, -0.625)
    # a coefficient (deterministic, eta-independent) off by ~4e-6 must still fail
    actual = _fit_gls_like(0.25, -0.625 * (1 + 4e-6))
    with pytest.raises(AssertionError):
        assert_matches(actual, expected, loose_keys=loose)


def test_only_the_fit_gls_fixture_has_loose_keys() -> None:
    assert loose_keys_for("fit_gls")
    for other in ("fit", "anova", "optimal", "diagnostics", "coverage", "desirability"):
        assert loose_keys_for(other) == frozenset()
