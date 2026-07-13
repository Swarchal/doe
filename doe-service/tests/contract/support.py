"""Shared plumbing for the contract tests: pair loading + structural comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PAIRS_DIR = Path(__file__).parent / "pairs"


def load_pair(name: str) -> dict[str, Any]:
    """Load one ``pairs/<name>.json`` golden request/response fixture."""
    return json.loads((PAIRS_DIR / f"{name}.json").read_text())


def pair_names() -> list[str]:
    """Every fixture's name (its file stem), for parametrizing the "every pair loads
    and structurally matches" smoke test."""
    return sorted(p.stem for p in PAIRS_DIR.glob("*.json"))


def assert_matches(actual: Any, expected: Any, *, path: str = "$") -> None:
    """Structural comparison: exact for ``str``/``bool``/``None``, ``pytest.approx`` for
    numbers, recursive for ``dict``/``list`` (same key set, same length, matched
    positionally). This is the "not brittle float-exact where the library uses
    search/RNG" anchor from ``docs/WEBSERVICE_BUILD.md`` §6 -- every number in a golden
    fixture is checked, just not for bit-exact equality.
    """
    # bool is an int subclass, so this check must come first.
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, int | float) and isinstance(actual, int | float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), (
            f"{path}: {actual!r} != approx({expected!r})"
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert actual.keys() == expected.keys(), (
            f"{path}: key set {sorted(actual)} != {sorted(expected)}"
        )
        for key in expected:
            assert_matches(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_matches(a, e, path=f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
