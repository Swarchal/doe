"""Screening design generators (Phase 5a).

Implemented:
    * :func:`definitive_screening` -- Jones-Nachtsheim conference-matrix DSDs

A definitive screening design (DSD) screens ``k`` factors' main effects *and* detects
curvature / two-factor interactions in ``2k + 1`` runs, collapsing the classic
full-factorial -> CCD two-stage screen into one three-level design. See ``docs/PHASE5.md``
section 1 for the build plan and correctness anchors.

Analysis is unchanged: a DSD is a plain :class:`~doe.design.Design`; because it uses three
coded levels ``{-1, 0, +1}``, ``build_model_matrix`` already emits squared terms for its
continuous factors, so ``fit_ols`` / ``anova_table`` / ``half_normal_plot`` consume it as-is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..design import Design
from ..factors import CategoricalFactor, Factor, FactorSet
from .factorial import _decode, _generator_spec, _require_box_factors

#: How far past the requested order to search when reporting the nearest constructible
#: conference-matrix size in an error message.
_SEARCH_SPAN = 64


def _is_prime(n: int) -> bool:
    """Simple trial-division primality test (only ever called on small screening sizes)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_power_factor(q: int) -> tuple[int, int] | None:
    """Return ``(p, m)`` if ``q == p**m`` for a prime ``p``, else ``None``."""
    if q < 2:
        return None
    p = None
    limit = math.isqrt(q)
    for candidate in range(2, limit + 1):
        if q % candidate == 0:
            p = candidate
            break
    if p is None:
        p = q  # q has no factor <= sqrt(q), so q itself is prime
    if not _is_prime(p):
        return None
    m = 0
    remaining = q
    while remaining % p == 0:
        remaining //= p
        m += 1
    return (p, m) if remaining == 1 else None


def _jacobsthal_matrix(p: int, m: int) -> np.ndarray:
    """The ``q x q`` Jacobsthal (quadratic-character) matrix over ``GF(p**m)``, ``q = p**m``.

    ``Q[i, j] = chi(e_i - e_j)`` where ``chi`` is the quadratic character (``0`` at zero,
    ``+1``/``-1`` according to whether the field element is a nonzero square) and ``e``
    enumerates the field elements. ``Q`` has zero diagonal and ``+/-1`` off it, and satisfies
    ``Q Q^T = q I - J`` for any prime power ``q`` -- the identity the Paley border construction
    in :func:`_conference_matrix` relies on.

    Only ``m in {1, 2}`` are implemented: ``m=1`` is plain arithmetic mod ``p``; ``m=2`` builds
    ``GF(p**2)`` as ``{a + b*x : a, b in Z_p}`` with ``x**2`` fixed to a quadratic non-residue
    mod ``p`` (so ``x**2 - nonresidue`` is irreducible over ``GF(p)``). Cubic and higher
    extensions (e.g. ``p**3 = 27``) are not supported.
    """
    if m == 1:

        def is_square(a: int) -> bool:
            return pow(a, (p - 1) // 2, p) == 1

        def chi1(a: int) -> float:
            a %= p
            if a == 0:
                return 0.0
            return 1.0 if is_square(a) else -1.0

        elements1 = list(range(p))
        q = p
        Q = np.zeros((q, q))
        for i, ei in enumerate(elements1):
            for j, ej in enumerate(elements1):
                Q[i, j] = chi1(ei - ej)
        return Q

    if m == 2:
        nonresidue = next(b for b in range(1, p) if pow(b, (p - 1) // 2, p) != 1)

        def mul(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
            a, b = x
            c, d = y
            return ((a * c + b * d * nonresidue) % p, (a * d + b * c) % p)

        def sub(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
            return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)

        elements2 = [(a, b) for a in range(p) for b in range(p)]
        squares = {mul(e, e) for e in elements2 if e != (0, 0)}

        def chi2(e: tuple[int, int]) -> float:
            if e == (0, 0):
                return 0.0
            return 1.0 if e in squares else -1.0

        q = p * p
        Q = np.zeros((q, q))
        for i, gi in enumerate(elements2):
            for j, gj in enumerate(elements2):
                Q[i, j] = chi2(sub(gi, gj))
        return Q

    raise ValueError(f"unsupported Galois-field extension degree m={m}")


def _constructible_order(order: int) -> bool:
    """True if :func:`_conference_matrix` can build this (even) order."""
    if order < 2 or order % 2 != 0:
        return False
    if order == 2:
        return True
    factor = _prime_power_factor(order - 1)
    return factor is not None and factor[1] <= 2


def _nearest_constructible_orders(order: int) -> list[int]:
    """Constructible even orders near ``order``, for a helpful error message."""
    return [n for n in range(2, order + _SEARCH_SPAN, 2) if _constructible_order(n)]


def _conference_matrix(order: int) -> np.ndarray:
    """Return a conference matrix of the given (even) ``order``.

    A conference matrix ``C`` is ``order x order`` with zero diagonal, ``+/-1`` off the
    diagonal, and ``Cᵀ C = (order - 1) I`` (mutually orthogonal columns).

    Built via the Paley border construction: for ``q = order - 1`` a prime or odd-prime-square,
    ``C = [[0, 1ᵀ], [1, Q]]`` where ``Q`` is the ``q x q`` Jacobsthal matrix over ``GF(q)``
    (see :func:`_jacobsthal_matrix`). This covers every even order whose predecessor is prime
    or the square of an odd prime (e.g. 4, 6, 8, 10, 12, 14, 18, 20, 24, 26, 30, ...); orders
    like 16, 22, 28 (predecessor ``15 = 3*5``, ``21 = 3*7``, ``27 = 3**3``) are not covered and
    raise ``ValueError`` naming the nearest constructible sizes. Order 2 is the trivial
    ``[[0, 1], [1, 0]]`` conference matrix.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"conference matrix order must be a positive even integer, got {order}")
    if order == 2:
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    q = order - 1
    factor = _prime_power_factor(q)
    if factor is None or factor[1] > 2:
        nearby = _nearest_constructible_orders(order)
        lower = [n for n in nearby if n < order]
        upper = [n for n in nearby if n > order]
        hints = []
        if lower:
            hints.append(f"{lower[-1]} (smaller)")
        if upper:
            hints.append(f"{upper[0]} (larger)")
        raise ValueError(
            f"no conference matrix construction available for order {order} "
            f"({q} is not prime or an odd prime square); "
            f"nearest constructible order(s): {', '.join(hints) if hints else 'none nearby'}"
        )

    p, m = factor
    Q = _jacobsthal_matrix(p, m)
    C = np.zeros((order, order))
    C[0, 1:] = 1.0
    C[1:, 0] = 1.0
    C[1:, 1:] = Q
    return C


def definitive_screening(
    factors: Sequence[Factor],
    *,
    extra_center_runs: int = 0,
    fake_factors: int | None = None,
) -> Design:
    """Generate a definitive screening design (Jones & Nachtsheim, 2011).

    A DSD estimates the ``k`` main effects of ``k`` continuous factors, is orthogonal for
    main effects (main effects are uncorrelated with each other *and* with every
    second-order term), and -- because every factor runs at three levels -- makes curvature
    estimable, all in ``2k + 1`` runs. The design is the row stack ``[C; -C; 0ᵀ]``: a
    conference matrix ``C`` of order ``k``, its foldover ``-C``, and one all-zero center run.

    For an odd number of factors, one **fake factor** is added to reach an even
    conference-matrix order and then dropped (yielding ``2k + 3`` runs); ``fake_factors``
    overrides the count, ``None`` auto-adds the minimum (1 iff ``k`` is odd, else 0). Because
    every pair of columns in a conference matrix is mutually orthogonal by construction,
    dropping the fake column(s) after generation leaves the real factors' orthogonality intact.

    Args:
        factors: the continuous factors to screen (``k >= 3`` of them). Categorical factors
            are rejected -- the Jones-Nachtsheim (2013) two-level categorical extension is not
            implemented; use :func:`~doe.generators.optimal.d_optimal` instead.
        extra_center_runs: additional all-zero center runs appended beyond the single
            structural one (for a purer lack-of-fit pure-error estimate).
        fake_factors: number of fake (dropped) factors to pad with; ``None`` auto-adds one
            iff the factor count is odd.

    Returns:
        A :class:`~doe.design.Design` in natural units with coded levels ``{-1, 0, +1}``.
        The structural center run (and any ``extra_center_runs``) is tagged ``"center"`` in
        ``point_types``, the remaining runs are tagged ``"dsd"``; ``meta["generator"]``
        records the call for regeneration and ``meta["fake_factors"]`` the resolved count.

    Raises:
        ValueError: for fewer than 3 factors, a negative ``extra_center_runs``/``fake_factors``,
            an odd real+fake factor count, categorical factors, or an unconstructible
            conference-matrix order (message names the nearest sizes that do work).
    """
    fs = FactorSet(factors)
    _require_box_factors(fs)
    categorical = [f.name for f in fs if isinstance(f, CategoricalFactor)]
    if categorical:
        raise ValueError(
            f"definitive_screening does not support categorical factors {categorical} "
            "(the Jones-Nachtsheim 2013 two-level categorical extension is not implemented); "
            "use doe.generators.optimal.d_optimal instead"
        )

    k = len(fs)
    if k < 3:
        raise ValueError("a definitive screening design needs at least 3 factors")
    if extra_center_runs < 0:
        raise ValueError("extra_center_runs must be >= 0")

    if fake_factors is None:
        n_fake = 1 if k % 2 == 1 else 0
    else:
        if fake_factors < 0:
            raise ValueError("fake_factors must be >= 0")
        n_fake = fake_factors

    order = k + n_fake
    if order % 2 != 0:
        raise ValueError(
            f"fake_factors={fake_factors} leaves an odd conference-matrix order {order}; "
            "the real + fake factor count must be even"
        )

    C = _conference_matrix(order)
    structural = np.vstack([C, -C])  # the foldover [C; -C]
    coded = structural[:, :k]  # drop the trailing fake-factor column(s), if any
    coded = np.vstack([coded, np.zeros((1, k))])  # the structural all-zero center run
    if extra_center_runs:
        coded = np.vstack([coded, np.zeros((extra_center_runs, k))])

    point_types = ["dsd"] * (2 * order) + ["center"] * (1 + extra_center_runs)
    runs = _decode(fs, coded)
    meta: dict[str, object] = {
        "generator": _generator_spec(
            "definitive_screening",
            extra_center_runs=extra_center_runs,
            fake_factors=fake_factors,
        ),
        "fake_factors": n_fake,
    }
    return Design(
        runs,
        fs,
        name=f"definitive_screening_k{k}",
        meta=meta,
        point_types=tuple(point_types),
    )
