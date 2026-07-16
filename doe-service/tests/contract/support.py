"""Shared plumbing for the contract tests: pair loading + structural comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PAIRS_DIR = Path(__file__).parent / "pairs"

#: Default numeric tolerance: tight, so a regression that shifts a *deterministic* fitted
#: coefficient / ANOVA SS / design metric by even ~1e-5 is caught.
_TIGHT_REL = 1e-6

#: Looser tolerance for values a *search* derives, which are reproducible only to the
#: search's own precision (not machine precision), so they vary by ~1e-5 relative across
#: platforms' BLAS/LAPACK builds.
_SEARCH_REL = 1e-5

#: Leaf keys whose value a search derives, scoped **per fixture** so the looseness never
#: weakens a deterministic value. The only affected fixture is the split-plot REML fit: it
#: profiles a *flat* likelihood over the whole-plot variance ratio η, whose argmin the last
#: ULP of the BLAS/LAPACK log-det/solve fixes (tightening the optimiser tolerance does *not*
#: help -- the flat well makes the argmin intrinsically noise-limited). So η -- and every GLS
#: quantity derived from it: standard errors, t/p-values, confidence bounds, and σ²_wp / mse
#: themselves -- moves ~1e-5 between platforms, while the fitted coefficients, effects, fitted
#: values and R² stay η-independent (GLS ≈ OLS for the balanced means) and locked at _TIGHT_REL.
_SEARCH_DERIVED_KEYS_BY_PAIR: dict[str, frozenset[str]] = {
    "fit_gls": frozenset({"std_error", "t", "p", "ci_low", "ci_high", "sigma2_wp", "mse"}),
}


def load_pair(name: str) -> dict[str, Any]:
    """Load one ``pairs/<name>.json`` golden request/response fixture."""
    return json.loads((PAIRS_DIR / f"{name}.json").read_text())


def pair_names() -> list[str]:
    """Every fixture's name (its file stem), for parametrizing the "every pair loads
    and structurally matches" smoke test."""
    return sorted(p.stem for p in PAIRS_DIR.glob("*.json"))


def loose_keys_for(name: str) -> frozenset[str]:
    """The search-derived leaf keys for fixture ``name`` (empty for the deterministic ones)."""
    return _SEARCH_DERIVED_KEYS_BY_PAIR.get(name, frozenset())


def assert_matches(
    actual: Any,
    expected: Any,
    *,
    path: str = "$",
    loose_keys: frozenset[str] = frozenset(),
    rel: float = _TIGHT_REL,
) -> None:
    """Structural comparison: exact for ``str``/``bool``/``None``, ``pytest.approx`` for
    numbers, recursive for ``dict``/``list`` (same key set, same length, matched
    positionally). This is the "not brittle float-exact where the library uses
    search/RNG" anchor from ``docs/WEBSERVICE_BUILD.md`` §6 -- every number in a golden
    fixture is checked, just not for bit-exact equality.

    Numbers are compared at ``rel`` (``_TIGHT_REL`` by default); a leaf whose key is in
    ``loose_keys`` -- the search-derived quantities of a fixture (see
    :data:`_SEARCH_DERIVED_KEYS_BY_PAIR`) -- and everything under it drop to the looser
    ``_SEARCH_REL``, so BLAS/LAPACK noise in those values does not make the check flaky
    while every deterministic value stays locked tight.
    """
    # bool is an int subclass, so this check must come first.
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, int | float) and isinstance(actual, int | float):
        assert actual == pytest.approx(expected, rel=rel, abs=1e-9), (
            f"{path}: {actual!r} != approx({expected!r}, rel={rel})"
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert actual.keys() == expected.keys(), (
            f"{path}: key set {sorted(actual)} != {sorted(expected)}"
        )
        for key in expected:
            child_rel = _SEARCH_REL if key in loose_keys else rel
            assert_matches(
                actual[key], expected[key], path=f"{path}.{key}",
                loose_keys=loose_keys, rel=child_rel,
            )
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_matches(a, e, path=f"{path}[{i}]", loose_keys=loose_keys, rel=rel)
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
