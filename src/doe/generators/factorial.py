"""Factorial design generators (Phase 1).

Implemented:
    * :func:`full_factorial`       -- general full factorial in coded levels
    * :func:`fractional_factorial` -- 2-level fractions from generator strings
    * :func:`plackett_burman`      -- saturated main-effect screening designs
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.linalg import hankel, toeplitz

from ..design import Design
from ..factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet


def _generator_spec(name: str, **parameters: object) -> dict[str, object]:
    """The ``meta["generator"]`` block: the call that regenerates a design.

    Records the generator's public name and its *requested* parameters (values resolved
    from the request, e.g. a numeric alpha, belong as ordinary ``meta`` keys instead).
    This is what lets a serialized design reconstruct the intended experiment rather
    than only replay the frozen run table. Values must be JSON-ready: ``Design.to_dict``
    does not coerce inside nested containers.
    """
    return {"name": name, "parameters": parameters}


def _coded_levels(n_levels: int) -> np.ndarray:
    """Evenly spaced coded levels in ``[-1, +1]`` (e.g. 2 -> [-1, 1], 3 -> [-1, 0, 1])."""
    if n_levels < 2:
        raise ValueError("a factor needs at least 2 levels")
    return np.linspace(-1.0, 1.0, n_levels)


def _require_box_factors(factors: FactorSet) -> None:
    """Reject mixture components -- factorial recipes cover a box, not the simplex."""
    mixture = [f.name for f in factors if not isinstance(f, ContinuousFactor | CategoricalFactor)]
    if mixture:
        raise TypeError(
            f"factorial designs do not support mixture components {mixture}; "
            "use the generators in doe.generators.mixture"
        )


def _decode(factors: FactorSet, coded: np.ndarray) -> pd.DataFrame:
    """Turn a coded design matrix into a natural-unit runs frame."""
    cols: dict[str, np.ndarray | list[object]] = {}
    for j, factor in enumerate(factors):
        if isinstance(factor, ContinuousFactor):
            cols[factor.name] = factor.decode(coded[:, j])
        elif isinstance(factor, CategoricalFactor):
            idx = ((coded[:, j] + 1) / 2 * (len(factor.levels) - 1)).round().astype(int)
            cols[factor.name] = [factor.levels[int(i)] for i in idx]
        else:  # pragma: no cover - _require_box_factors rejects these up front
            raise TypeError(f"unsupported factor type for {factor!r}")
    return pd.DataFrame(cols)


def _validate_two_level_categoricals(factors: FactorSet) -> None:
    """Reject categorical factors that a strictly two-level design cannot encode."""
    _require_box_factors(factors)
    bad = [
        factor.name
        for factor in factors
        if isinstance(factor, CategoricalFactor) and len(factor.levels) > 2
    ]
    if bad:
        raise ValueError(
            "two-level designs cannot encode categorical factors with >2 levels: "
            f"{bad}; use full_factorial or split the factor"
        )


def full_factorial(factors: Sequence[Factor], levels: int | Sequence[int] = 2) -> Design:
    """Generate a full factorial design.

    Args:
        factors: the factors to vary.
        levels: a single level count applied to every factor, or one count per factor.
            Categorical factors always use their own number of levels.
    """
    fs = FactorSet(factors)
    _require_box_factors(fs)
    if isinstance(levels, int):
        per = [levels] * len(fs)
    else:
        per = list(levels)
        if len(per) != len(fs):
            raise ValueError("levels sequence length must match number of factors")
    per = [
        len(f.levels) if isinstance(f, CategoricalFactor) else int(n)
        for f, n in zip(fs, per, strict=True)
    ]

    grids = [_coded_levels(n) for n in per]
    coded = np.array(list(itertools.product(*grids)), dtype=float)
    runs = _decode(fs, coded)
    return Design(
        runs,
        fs,
        name=f"full_factorial_{'x'.join(map(str, per))}",
        meta={"generator": _generator_spec("full_factorial", levels=per)},
    )


def fractional_factorial(factors: Sequence[Factor], generators: Sequence[str]) -> Design:
    """Generate a 2-level fractional factorial from generator strings.

    A fractional factorial trades runs for resolution: rather than every corner of the
    factor cube, it runs a carefully chosen fraction. The price is *aliasing* -- each
    generator deliberately sets a new factor equal to an interaction of the base factors, so
    that factor's main effect and that interaction are confounded and cannot be told apart.
    The choice of generators determines the design's resolution (how high-order the
    confounded terms are); a good generator aliases new main effects only with high-order
    interactions that are assumed negligible.

    Each generator names an extra factor in terms of a product of base factors, e.g.
    ``"D=ABC"`` aliases factor D with the ABC interaction. Base factors are the first
    ``len(factors) - len(generators)`` entries of ``factors``.

    Example:
        A 2^(4-1) design::

            fractional_factorial([A, B, C, D], generators=["D=ABC"])
    """
    fs = FactorSet(factors)
    _validate_two_level_categoricals(fs)
    n_base = len(fs) - len(generators)
    if n_base < 1:
        raise ValueError("need at least one base factor")

    coded = np.array(list(itertools.product([-1.0, 1.0], repeat=n_base)), dtype=float)
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

        # build the generated factor's column as the elementwise product of its base columns:
        # this *is* the aliasing relation (e.g. D = A*B*C), so column D and the ABC interaction
        # are numerically identical and their effects are confounded by construction.
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
    return Design(
        runs,
        fs,
        name=f"fractional_factorial_{len(fs)}-{len(generators)}",
        # the generator strings are the defining relation -- the design's alias structure
        # is unrecoverable from the run table alone, so they must survive serialization.
        meta={"generator": _generator_spec("fractional_factorial", generators=list(generators))},
    )


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


# Cyclic generating vectors for the Plackett-Burman base Hadamard matrices that are
# not simple powers of two (Plackett & Burman, 1946). Each builds an order-``base``
# Hadamard matrix with a leading row/column of +1; the factor columns come from the
# bordered ``toeplitz``/``hankel`` core. Larger sizes are obtained by Sylvester doubling.
_PB_BASES: tuple[int, ...] = (1, 12, 20)


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _pb_constructible(n: int) -> bool:
    """True if an order-``n`` Plackett-Burman matrix is available (``base * 2**e``)."""
    return any(n % base == 0 and _is_power_of_two(n // base) for base in _PB_BASES)


def _pb_size(k: int) -> int:
    """Smallest constructible run count ``n`` (a multiple of 4) with ``n >= k + 1``."""
    n = 4 * math.ceil((k + 1) / 4)
    while not _pb_constructible(n):
        n += 4
    return n


def _pb_base_matrix(base: int) -> np.ndarray:
    """The order-``base`` Hadamard matrix with a leading +1 border."""
    if base == 1:
        return np.ones((1, 1))
    if base == 12:
        first_col = [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1]
        first_row = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
        core = toeplitz(first_col, first_row)
    elif base == 20:
        first_col = [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1]
        first_row = [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1]
        core = hankel(first_col, first_row)
    else:  # pragma: no cover - guarded by _pb_constructible
        raise ValueError(f"no Plackett-Burman base matrix for size {base}")

    matrix = np.ones((base, base))
    matrix[1:, 1:] = core
    return matrix


def _pb_matrix(n: int) -> np.ndarray:
    """An order-``n`` Plackett-Burman (Hadamard) matrix with a leading +1 column."""
    for base in _PB_BASES:
        if n % base == 0 and _is_power_of_two(n // base):
            matrix = _pb_base_matrix(base)
            doublings = (n // base).bit_length() - 1  # log2(n / base)
            for _ in range(doublings):
                matrix = np.block([[matrix, matrix], [matrix, -matrix]])
            return matrix
    raise ValueError(f"no Plackett-Burman construction for size {n}")


def plackett_burman(factors: Sequence[Factor]) -> Design:
    """Generate a 2-level Plackett-Burman screening design.

    Plackett-Burman designs estimate the ``k`` main effects of ``k`` factors in ``n``
    runs, where ``n`` is the smallest available multiple of four with ``n >= k + 1``
    (e.g. 7 factors in 8 runs, 11 in 12, 19 in 20). They are saturated, orthogonal
    main-effect designs: every factor column is balanced and mutually orthogonal, but
    two-factor interactions are *partially* aliased with main effects, so they screen
    which factors matter rather than resolving interactions.

    Run counts come from Sylvester doubling of the base Hadamard matrices of order 1
    (powers of two), 12 and 20. Sizes needing other base constructions (e.g. 28, 36)
    are skipped in favour of the next larger available size.
    """
    fs = FactorSet(factors)
    _validate_two_level_categoricals(fs)
    k = len(fs)
    if k < 1:
        raise ValueError("need at least one factor")

    n = _pb_size(k)
    matrix = _pb_matrix(n)
    coded = matrix[:, 1 : k + 1]  # drop the all-+1 leading column; keep k factor columns
    runs = _decode(fs, coded)
    return Design(
        runs,
        fs,
        name=f"plackett_burman_{n}",
        meta={"generator": _generator_spec("plackett_burman")},
    )
