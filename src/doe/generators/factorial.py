"""Factorial design generators (Phase 1).

Implemented:
    * :func:`full_factorial`       -- general full factorial in coded levels
    * :func:`fractional_factorial` -- 2-level fractions from generator strings

Stubbed (later phases):
    * :func:`plackett_burman`
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import ContinuousFactor, Factor, FactorSet


def _coded_levels(n_levels: int) -> np.ndarray:
    """Evenly spaced coded levels in ``[-1, +1]`` (e.g. 2 -> [-1, 1], 3 -> [-1, 0, 1])."""
    if n_levels < 2:
        raise ValueError("a factor needs at least 2 levels")
    return np.linspace(-1.0, 1.0, n_levels)


def _decode(factors: FactorSet, coded: np.ndarray) -> pd.DataFrame:
    """Turn a coded design matrix into a natural-unit runs frame."""
    cols: dict[str, np.ndarray | list[object]] = {}
    for j, factor in enumerate(factors):
        if isinstance(factor, ContinuousFactor):
            cols[factor.name] = factor.decode(coded[:, j])
        else:
            idx = ((coded[:, j] + 1) / 2 * (len(factor.levels) - 1)).round().astype(int)
            cols[factor.name] = [factor.levels[int(i)] for i in idx]
    return pd.DataFrame(cols)


def full_factorial(factors: Sequence[Factor], levels: int | Sequence[int] = 2) -> Design:
    """Generate a full factorial design.

    Args:
        factors: the factors to vary.
        levels: a single level count applied to every factor, or one count per factor.
            Categorical factors always use their own number of levels.
    """
    fs = FactorSet(factors)
    if isinstance(levels, int):
        per = [levels] * len(fs)
    else:
        per = list(levels)
        if len(per) != len(fs):
            raise ValueError("levels sequence length must match number of factors")
    per = [
        len(f.levels) if not isinstance(f, ContinuousFactor) else n
        for f, n in zip(fs, per, strict=True)
    ]

    grids = [_coded_levels(n) for n in per]
    coded = np.array(list(itertools.product(*grids)), dtype=float)
    runs = _decode(fs, coded)
    return Design(runs, fs, name=f"full_factorial_{'x'.join(map(str, per))}")


def fractional_factorial(factors: Sequence[Factor], generators: Sequence[str]) -> Design:
    """Generate a 2-level fractional factorial from generator strings.

    Each generator names an extra factor in terms of a product of base factors, e.g.
    ``"D=ABC"`` aliases factor D with the ABC interaction. Base factors are the first
    ``len(factors) - len(generators)`` entries of ``factors``.

    Example:
        A 2^(4-1) design::

            fractional_factorial([A, B, C, D], generators=["D=ABC"])
    """
    fs = FactorSet(factors)
    n_base = len(fs) - len(generators)
    if n_base < 1:
        raise ValueError("need at least one base factor")

    base = full_factorial([f for f in list(fs)[:n_base]], levels=2)
    coded = base.coded().to_numpy()
    base_names = fs.names[:n_base]
    generated_names = fs.names[n_base:]

    generated_cols: dict[str, np.ndarray] = {}
    for gen in generators:
        if gen.count("=") != 1:
            raise ValueError(f"malformed generator {gen!r}; expected e.g. 'D=ABC'")
        lhs, rhs = (part.strip() for part in gen.split("=", maxsplit=1))
        if not lhs or not rhs:
            raise ValueError(f"malformed generator {gen!r}; expected e.g. 'D=ABC'")

        try:
            target = _generator_factor_name(lhs, fs.names)
        except ValueError as exc:
            raise ValueError(
                f"generator {gen!r} left-hand side must name one of the generated "
                f"factors {generated_names}"
            ) from exc
        if target not in generated_names:
            raise ValueError(
                f"generator {gen!r} left-hand side must name one of the generated "
                f"factors {generated_names}"
            )
        if target in generated_cols:
            raise ValueError(f"duplicate generator for factor {target!r}")

        col = np.ones(coded.shape[0])
        for letter in rhs:
            try:
                col = col * coded[:, base_names.index(_letter_to_name(letter, base_names))]
            except ValueError as exc:
                raise ValueError(f"generator {gen!r} references unknown factor {letter!r}") from exc
        generated_cols[target] = col

    extra_cols = [generated_cols[name] for name in generated_names]
    full = np.column_stack([coded, *extra_cols]) if extra_cols else coded
    runs = _decode(fs, full)
    return Design(runs, fs, name=f"fractional_factorial_{len(fs)}-{len(generators)}")


def _letter_to_name(letter: str, base_names: list[str]) -> str:
    """Map a positional generator letter (A, B, C, ...) to a base factor name."""
    pos = ord(letter.upper()) - ord("A")
    if 0 <= pos < len(base_names):
        return base_names[pos]
    raise ValueError(letter)


def _generator_factor_name(token: str, factor_names: list[str]) -> str:
    """Resolve a generator target by exact factor name or positional letter."""
    if token in factor_names:
        return token
    if len(token) == 1 and token.isalpha():
        pos = ord(token.upper()) - ord("A")
        if 0 <= pos < len(factor_names):
            return factor_names[pos]
    raise ValueError(token)


def plackett_burman(factors: Sequence[Factor]) -> Design:
    """Plackett-Burman screening design. Not yet implemented (Phase 1 TODO)."""
    raise NotImplementedError("plackett_burman is planned for Phase 1 completion")
