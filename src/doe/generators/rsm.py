"""Response-surface design generators.

Second-order designs that let us fit curvature:

    * :func:`central_composite` -- factorial core + axial/star points + center points
    * :func:`box_behnken`       -- spherical 3-level design, no corner (extreme) runs
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import ContinuousFactor, Factor, FactorSet
from .factorial import _generator_spec, fractional_factorial

#: How far the axial/star points sit from the center, in coded units.
AlphaSpec = float | Literal["faced", "rotatable", "orthogonal"]


def _require_continuous(fs: FactorSet) -> None:
    bad = [f.name for f in fs if not isinstance(f, ContinuousFactor)]
    if bad:
        raise ValueError(
            f"response-surface designs require continuous factors; got non-continuous: {bad}"
        )


def _decode_coded(fs: FactorSet, coded: np.ndarray) -> pd.DataFrame:
    """Map a coded design matrix (all-continuous factors) to a natural-unit runs frame."""
    cols: dict[str, np.ndarray] = {}
    for j, factor in enumerate(fs):
        assert isinstance(factor, ContinuousFactor)  # guarded by _require_continuous
        cols[factor.name] = factor.decode(coded[:, j])
    return pd.DataFrame(cols)


def _resolve_alpha(alpha: AlphaSpec, n_factorial: int, k: int, center: int) -> float:
    """Turn an alpha spec into a numeric axial distance in coded units."""
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        return float(alpha)
    if alpha == "faced":
        # axial points on the cube faces (alpha = 1) -- keeps every run within the stated
        # low/high bounds, at the cost of rotatability. Use when factor levels are hard limits.
        return 1.0
    if alpha == "rotatable":
        # alpha = (n_factorial)^(1/4) makes prediction variance depend only on distance from the
        # center, so the model predicts equally well in every direction -- the conventional RSM
        # textbook choice (this library nonetheless defaults to "faced" to stay within bounds).
        return float(n_factorial**0.25)
    if alpha == "orthogonal":
        # value that makes the second-order terms uncorrelated (NIST e-Handbook form)
        n_total = n_factorial + 2 * k + center
        return float((((n_factorial * n_total) ** 0.5 - n_factorial) / 2.0) ** 0.5)
    raise ValueError(f"unknown alpha spec {alpha!r}")


def central_composite(
    factors: Sequence[Factor],
    *,
    alpha: AlphaSpec = "faced",
    center: int = 4,
    fraction: Sequence[str] | None = None,
) -> Design:
    """Generate a central composite design (CCD).

    A CCD is a 2-level factorial (or resolution-V fraction) *core* at coded ``+/-1``,
    plus ``2k`` *axial* runs at ``+/-alpha`` on each axis (others at 0), plus ``center``
    replicated *center* runs at the origin.

    Args:
        factors: the continuous factors to vary (``k`` of them).
        alpha: axial distance. ``"faced"`` (default) uses ``alpha=1`` so every run stays
            inside the stated ``low``/``high`` bounds (face-centered, CCF). ``"rotatable"``
            uses ``(n_factorial)**0.25`` and ``"orthogonal"`` solves for uncorrelated
            quadratic terms -- both place axial points *outside* ``[-1, +1]`` and therefore
            decode beyond the stated bounds. A float sets ``alpha`` explicitly.
        center: number of center-point replicates (pure-error source for lack-of-fit).
        fraction: optional generator strings (e.g. ``["E=ABCD"]``) for a fractional core.

    The returned design records ``meta["generator"]`` (the requested ``alpha``/``center``/
    ``fraction``, so a serialized design can be regenerated), plus ``meta["alpha"]`` (the
    resolved numeric axial distance) and ``meta["axial_extrapolates"]`` (``True`` when axial
    points fall outside the bounds).
    """
    fs = FactorSet(factors)
    _require_continuous(fs)
    k = len(fs)
    if k < 2:
        raise ValueError("a central composite design needs at least 2 factors")

    if fraction is None:
        core = np.array(list(itertools.product([-1.0, 1.0], repeat=k)), dtype=float)
    else:
        core = fractional_factorial(list(fs), fraction).coded().to_numpy(dtype=float)
    n_factorial = core.shape[0]

    a = _resolve_alpha(alpha, n_factorial, k, center)

    # Axial (star) points sit out along each factor axis with all other factors at zero. They
    # are what gives the design a third level per factor, so the pure quadratic (curvature)
    # terms become estimable -- a 2-level factorial alone can only fit a plane. ``alpha`` sets
    # how far out they reach, trading off rotatability vs. staying inside the factor bounds.
    axial = np.zeros((2 * k, k), dtype=float)
    for j in range(k):
        axial[2 * j, j] = -a
        axial[2 * j + 1, j] = a

    # Replicated center points anchor the design at the origin: they supply the pure-error
    # estimate for lack-of-fit and let the fit detect overall curvature.
    center_block = np.zeros((center, k), dtype=float)

    coded = np.vstack([core, axial, center_block])
    point_types = ["factorial"] * n_factorial + ["axial"] * (2 * k) + ["center"] * center
    runs = _decode_coded(fs, coded)
    meta: dict[str, object] = {
        # the requested spec (e.g. alpha="rotatable") regenerates the design; the resolved
        # numeric alpha and extrapolation flag describe what that request produced.
        "generator": _generator_spec(
            "central_composite",
            alpha=alpha if isinstance(alpha, str) else float(alpha),
            center=center,
            fraction=list(fraction) if fraction is not None else None,
        ),
        "alpha": a,
        "axial_extrapolates": bool(a > 1.0 + 1e-9),
    }
    return Design(
        runs, fs, name=f"central_composite_k{k}", meta=meta, point_types=tuple(point_types)
    )


#: Largest ``k`` for which the all-pairs construction below coincides with the canonical
#: Box-Behnken design. For ``k in {3, 4, 5}`` every factor pair is used and the result is the
#: textbook rotatable BBD; from ``k = 6`` the canonical design uses only a balanced *subset* of
#: pairs (a BIBD), so taking all pairs would yield a larger, non-rotatable design under the
#: wrong name (e.g. k=6 -> 60 edge runs vs. the canonical 48).
_BOX_BEHNKEN_MAX_FACTORS = 5


def box_behnken(factors: Sequence[Factor], *, center: int = 3) -> Design:
    """Generate a Box-Behnken design (BBD).

    A spherical 3-level design with no corner runs: for each pair of factors, run the
    ``+/-1`` factorial of that pair with all other factors at 0, then add ``center``
    center-point replicates. Supports ``3 <= k <= 5`` factors.

    Because it never sets all factors to their extremes at once, a Box-Behnken design avoids
    the (often impossible or unsafe) corner combinations a central composite would visit, while
    still placing points at three levels so curvature is estimable. It is a good choice when the
    corners of the factor region are infeasible.

    Run-count anchors: ``k=3`` -> 12 edge + center; ``k=4`` -> 24 + center; ``k=5`` -> 40 +
    center.

    Note:
        Only ``k <= 5`` is supported. For ``k >= 6`` the canonical Box-Behnken design uses a
        balanced *incomplete* set of factor pairs rather than all of them, which this all-pairs
        construction does not reproduce; use :func:`central_composite` instead.
    """
    fs = FactorSet(factors)
    _require_continuous(fs)
    k = len(fs)
    if k < 3:
        raise ValueError("a Box-Behnken design needs at least 3 factors")
    if k > _BOX_BEHNKEN_MAX_FACTORS:
        raise ValueError(
            f"box_behnken supports at most {_BOX_BEHNKEN_MAX_FACTORS} factors "
            f"(got {k}): the canonical Box-Behnken design for k >= 6 uses a balanced subset "
            "of factor pairs that this all-pairs construction does not reproduce; "
            "use central_composite for more factors"
        )

    edges: list[np.ndarray] = []
    for i, j in itertools.combinations(range(k), 2):
        for vi, vj in itertools.product([-1.0, 1.0], repeat=2):
            row = np.zeros(k, dtype=float)
            row[i] = vi
            row[j] = vj
            edges.append(row)

    coded = np.vstack([np.array(edges), np.zeros((center, k), dtype=float)])
    point_types = ["edge"] * len(edges) + ["center"] * center
    runs = _decode_coded(fs, coded)
    return Design(
        runs,
        fs,
        name=f"box_behnken_k{k}",
        meta={"generator": _generator_spec("box_behnken", center=center)},
        point_types=tuple(point_types),
    )
