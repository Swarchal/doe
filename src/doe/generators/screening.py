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

Conference matrices are built, not tabulated: a Paley border over ``GF(p**m)`` plus a doubling
construction for the skew orders covers every even order from 2 to 32 except 22 -- and *no*
conference matrix of order 22 exists (see :func:`_order_exists`). Factor counts that land on
order 22 fall back to padding with fake factors.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from scipy.linalg import circulant

from ..design import Design
from ..factors import CategoricalFactor, Factor, FactorSet, MixtureFactor
from .factorial import _decode, _generator_spec

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


def _is_sum_of_two_squares(n: int) -> bool:
    """True if ``n == a**2 + b**2`` for non-negative integers ``a``, ``b``."""
    a = 0
    while a * a <= n:
        b = math.isqrt(n - a * a)
        if b * b + a * a == n:
            return True
        a += 1
    return False


def _poly_remainder(a: list[int], b: list[int], p: int) -> list[int]:
    """Remainder of polynomial ``a`` divided by monic polynomial ``b`` over ``GF(p)``.

    Coefficients are listed low-degree first, with trailing zeros stripped from the result
    (so the zero polynomial is ``[]``).
    """
    r = list(a)
    deg_b = len(b) - 1
    while True:
        while r and r[-1] == 0:
            r.pop()
        if len(r) - 1 < deg_b:
            return r
        shift = len(r) - 1 - deg_b
        lead = r[-1]
        for i, coeff in enumerate(b):
            r[i + shift] = (r[i + shift] - lead * coeff) % p


def _irreducible_poly(p: int, m: int) -> list[int]:
    """A monic degree-``m`` polynomial irreducible over ``GF(p)``, low-degree coefficients first.

    Found by trial division against every monic polynomial of degree ``1 .. m // 2`` -- a
    reducible polynomial always has a factor of at most half its degree. The search space is
    tiny for the field sizes screening designs reach (``GF(27)`` is the first one that needs
    this path at all).
    """
    for tail in itertools.product(range(p), repeat=m):
        if tail[0] == 0:
            continue  # divisible by x
        f = [*tail, 1]
        if all(
            _poly_remainder(f, [*divisor, 1], p)
            for degree in range(1, m // 2 + 1)
            for divisor in itertools.product(range(p), repeat=degree)
        ):
            return f
    raise ValueError(f"no irreducible polynomial of degree {m} over GF({p})")  # pragma: no cover


def _jacobsthal_matrix(p: int, m: int) -> np.ndarray:
    """The ``q x q`` Jacobsthal (quadratic-character) matrix over ``GF(p**m)``, ``q = p**m``.

    ``Q[i, j] = chi(e_i - e_j)`` where ``chi`` is the quadratic character (``0`` at zero,
    ``+1``/``-1`` according to whether the field element is a nonzero square) and ``e``
    enumerates the field elements. ``Q`` has zero diagonal and ``+/-1`` off it, and satisfies
    ``Q Q^T = q I - J`` for any prime power ``q`` -- the identity the Paley border construction
    in :func:`_conference_matrix` relies on.

    ``m=1`` is plain arithmetic mod ``p``. ``m=2`` builds ``GF(p**2)`` as
    ``{a + b*x : a, b in Z_p}`` with ``x**2`` fixed to a quadratic non-residue mod ``p`` (so
    ``x**2 - nonresidue`` is irreducible over ``GF(p)``); it keeps its own explicit binomial
    form so that the designs it already ships (orders 10 and 26) stay bit-for-bit stable.
    ``m>=3`` builds ``GF(p**m)`` generically, as degree-``m`` coefficient tuples multiplied
    modulo an irreducible polynomial from :func:`_irreducible_poly` -- this is the path that
    reaches ``GF(27)``, and with it conference-matrix order 28.
    """
    if m == 1:

        def is_square(a: int) -> bool:
            return pow(a, (p - 1) // 2, p) == 1

        def chi1(a: int) -> float:
            a %= p
            if a == 0:
                return 0.0
            return 1.0 if is_square(a) else -1.0

        # Q[i, j] = chi(i - j mod p) depends only on i - j, so Q is the circulant matrix whose
        # first column is the character vector [chi(0), chi(1), ..., chi(p-1)]
        # (scipy.linalg.circulant(c)[i, j] == c[(i - j) mod p]).
        return np.asarray(circulant(np.array([chi1(a) for a in range(p)])), dtype=float)

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

    poly = _irreducible_poly(p, m)
    zero = (0,) * m

    def mul_m(x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
        product = [0] * (2 * m - 1)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                product[i + j] = (product[i + j] + xi * yj) % p
        reduced = _poly_remainder(product, poly, p)
        return tuple(reduced + [0] * (m - len(reduced)))

    def sub_m(x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((a - b) % p for a, b in zip(x, y, strict=True))

    elements = list(itertools.product(range(p), repeat=m))
    squares_m = {mul_m(e, e) for e in elements if e != zero}

    q = p**m
    Qm = np.zeros((q, q))
    for i, ei in enumerate(elements):
        for j, ej in enumerate(elements):
            difference = sub_m(ei, ej)
            Qm[i, j] = 0.0 if difference == zero else (1.0 if difference in squares_m else -1.0)
    return Qm


def _skew_constructible_order(order: int) -> bool:
    """True if :func:`_skew_conference_matrix` can build this order.

    Skew conference matrices exist only for ``order % 4 == 0``. Two constructions reach them:
    the skew Paley border (``order - 1`` a prime power ``= 3 mod 4``, so its Jacobsthal matrix
    is skew), and doubling a smaller skew matrix.
    """
    if order < 4 or order % 4 != 0:
        return False
    factor = _prime_power_factor(order - 1)
    if factor is not None and (order - 1) % 4 == 3:
        return True
    return _skew_constructible_order(order // 2)


def _constructible_order(order: int) -> bool:
    """True if :func:`_conference_matrix` can build this (even) order."""
    if order < 2 or order % 2 != 0:
        return False
    if order == 2:
        return True
    return _prime_power_factor(order - 1) is not None or _skew_constructible_order(order)


def _order_exists(order: int) -> bool:
    """True unless a conference matrix of this even order is known *not* to exist.

    A conference matrix of order ``n = 2 mod 4`` must be symmetric, which forces ``n - 1`` to be
    a sum of two squares (Belevitch; van Lint & Seidel). Order 22 is the first casualty --
    ``21 = 3 * 7`` is not a sum of two squares -- so no amount of construction work will produce
    one. Orders ``n = 0 mod 4`` are only *conjectured* to always exist, so this returns ``True``
    for them: unconstructible there means "not implemented", not "impossible".
    """
    return order % 4 != 2 or _is_sum_of_two_squares(order - 1)


def _nearest_constructible_orders(order: int) -> list[int]:
    """Constructible even orders near ``order``, for a helpful error message."""
    return [n for n in range(2, order + _SEARCH_SPAN, 2) if _constructible_order(n)]


def _suggest_fake_factors(k: int, count: int = 2) -> list[int]:
    """The smallest ``fake_factors`` values giving ``k`` real factors a constructible order.

    Only values with the parity that keeps ``k + fake_factors`` even are returned (an odd
    conference-matrix order is never constructible), so these are directly usable suggestions
    for the :func:`definitive_screening` ``fake_factors`` argument.
    """
    start = k % 2  # 0 if k even, 1 if k odd -- keeps k + fake_factors even
    return [nf for nf in range(start, _SEARCH_SPAN, 2) if _constructible_order(k + nf)][:count]


def _skew_conference_matrix(order: int) -> np.ndarray:
    """Return a *skew* conference matrix (``Cᵀ = -C``) of the given ``order`` (a multiple of 4).

    Two constructions, tried in that order:

    * **Skew Paley border.** For ``q = order - 1`` a prime power ``= 3 mod 4``, the Jacobsthal
      matrix is skew (``chi(-1) = -1``), and bordering it with a *negated* first column,
      ``C = [[0, 1ᵀ], [-1, Q]]``, makes the whole matrix skew while keeping ``CᵀC = q I``.
    * **Doubling.** If ``C`` is skew of order ``n``, then ``[[C, C+I], [C-I, -C]]`` is skew of
      order ``2n``: the identity terms fill the diagonals of the off-diagonal blocks (which the
      zero diagonals of ``C`` would otherwise leave empty), and the cross terms cancel because
      ``Cᵀ = -C``. This is what reaches order 16, whose predecessor ``15 = 3 * 5`` is not a
      prime power, by doubling order 8.
    """
    factor = _prime_power_factor(order - 1)
    if factor is not None and (order - 1) % 4 == 3:
        p, m = factor
        C = np.zeros((order, order))
        C[0, 1:] = 1.0
        C[1:, 0] = -1.0
        C[1:, 1:] = _jacobsthal_matrix(p, m)
        return C

    half = _skew_conference_matrix(order // 2)
    identity = np.eye(order // 2)
    return np.block([[half, half + identity], [half - identity, -half]])


def _conference_matrix(order: int) -> np.ndarray:
    """Return a conference matrix of the given (even) ``order``.

    A conference matrix ``C`` is ``order x order`` with zero diagonal, ``+/-1`` off the
    diagonal, and ``Cᵀ C = (order - 1) I`` (mutually orthogonal columns).

    Built via the Paley border construction: for ``q = order - 1`` a prime power,
    ``C = [[0, 1ᵀ], [1, Q]]`` where ``Q`` is the ``q x q`` Jacobsthal matrix over ``GF(q)``
    (see :func:`_jacobsthal_matrix`); since ``Q Qᵀ = q I - J`` for *every* prime power, this
    border works whether ``Q`` is symmetric or skew. Orders whose predecessor is not a prime
    power (16, 22, 28 have predecessors ``15 = 3*5``, ``21 = 3*7``, and ``27 = 3**3`` -- the
    last of which *is* a prime power, reached through the ``GF(p**3)`` path) fall back to
    :func:`_skew_conference_matrix`, which reaches every multiple of 4 that doubling can build
    from a skew Paley matrix. Order 2 is the trivial ``[[0, 1], [1, 0]]`` conference matrix.

    The one gap in the range screening designs care about is **order 22, which does not exist**
    (see :func:`_order_exists`); unconstructible orders raise ``ValueError`` naming the nearest
    constructible sizes.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"conference matrix order must be a positive even integer, got {order}")
    if order == 2:
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    if not _constructible_order(order):
        nearby = _nearest_constructible_orders(order)
        lower = [n for n in nearby if n < order]
        upper = [n for n in nearby if n > order]
        hints = []
        if lower:
            hints.append(f"{lower[-1]} (smaller)")
        if upper:
            hints.append(f"{upper[0]} (larger)")
        reason = (
            f"no conference matrix of order {order} exists "
            f"({order} = 2 mod 4 requires {order - 1} to be a sum of two squares, and it is not)"
            if not _order_exists(order)
            else f"no conference matrix construction available for order {order}"
        )
        raise ValueError(
            f"{reason}; nearest constructible order(s): "
            f"{', '.join(hints) if hints else 'none nearby'}"
        )

    factor = _prime_power_factor(order - 1)
    if factor is None:
        return _skew_conference_matrix(order)

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

    When the requested factor count has no conference matrix of its own order, one or more
    **fake factors** are added to reach the next *constructible* order and then dropped (an odd
    ``k`` needs at least one; ``k = 21`` needs three, building at order 24, because no
    conference matrix of order 22 exists). ``fake_factors`` overrides the count; ``None``
    auto-adds the minimum that yields a constructible order.
    Because every pair of columns in a conference matrix is mutually orthogonal by construction,
    dropping the fake column(s) after generation leaves the real factors' orthogonality intact.

    Args:
        factors: the continuous factors to screen (``k >= 3`` of them). Categorical factors
            are rejected -- the Jones-Nachtsheim (2013) two-level categorical extension is not
            implemented; use :func:`~doe.generators.optimal.d_optimal` instead.
        extra_center_runs: additional all-zero center runs appended beyond the single
            structural one (for a purer lack-of-fit pure-error estimate).
        fake_factors: number of fake (dropped) factors to pad with; ``None`` auto-adds the
            minimum needed to reach a constructible conference-matrix order.

    Returns:
        A :class:`~doe.design.Design` in natural units with coded levels ``{-1, 0, +1}``.
        The structural center run (and any ``extra_center_runs``) is tagged ``"center"`` in
        ``point_types``, the remaining runs are tagged ``"dsd"``; ``meta["generator"]``
        records the call for regeneration and ``meta["fake_factors"]`` the resolved count.

    Raises:
        ValueError: for fewer than 3 factors, a negative ``extra_center_runs``/``fake_factors``,
            categorical or mixture factors, or an explicit ``fake_factors`` that leaves an
            unconstructible conference-matrix order (the message reports the shortfall in terms
            of ``k`` and suggests ``fake_factors`` values that do work). With ``fake_factors=None``
            the order is always constructible, so no ``ValueError`` for order is raised.
    """
    fs = FactorSet(factors)
    mixture = [f.name for f in fs if isinstance(f, MixtureFactor)]
    if mixture:
        raise ValueError(
            f"definitive_screening does not support mixture components {mixture}; "
            "a DSD screens a box region, not the simplex -- use the generators in "
            "doe.generators.mixture"
        )
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
        # Auto-pad to the next *constructible* conference-matrix order, not merely the next even
        # one: e.g. k=16 has no order-16 conference matrix, so add 2 fake factors and build at
        # order 18. n_fake advances by 2 to preserve the even order (parity matches k).
        n_fake = k % 2
        while not _constructible_order(k + n_fake):
            n_fake += 2
    else:
        if fake_factors < 0:
            raise ValueError("fake_factors must be >= 0")
        n_fake = fake_factors

    order = k + n_fake
    if not _constructible_order(order):
        suggestions = _suggest_fake_factors(k)
        hint = (
            "; try " + " or ".join(
                f"fake_factors={nf} ({2 * (k + nf) + 1} runs)" for nf in suggestions
            )
            if suggestions
            else ""
        )
        if order % 2 != 0:
            reason = f"order {order} is odd (the real + fake factor count must be even)"
        elif not _order_exists(order):
            reason = (
                f"no conference matrix of order {order} exists -- one would have to be "
                f"symmetric, which requires {order - 1} to be a sum of two squares"
            )
        else:
            reason = f"no conference matrix construction is available for order {order}"
        raise ValueError(
            f"definitive_screening cannot build a design for k={k} with "
            f"fake_factors={fake_factors}: {reason}{hint}"
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
