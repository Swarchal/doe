"""Mixture design generators.

Designs for formulation problems where the factors are *proportions of a whole* and every
run must sum to 1 -- the design region is a simplex, not a box:

    * :func:`simplex_lattice`     -- the ``{k, m}`` lattice over unconstrained components
    * :func:`simplex_centroid`    -- the ``2^k - 1`` subset centroids
    * :func:`extreme_vertices`    -- vertices (+ centroid) of a bound-constrained simplex
    * :func:`mixture_candidates`  -- a discrete candidate set feeding
      :func:`~doe.generators.optimal.coordinate_exchange` for D-optimal mixture designs

All generators require an all-:class:`~doe.factors.MixtureFactor` factor set and return a
:class:`~doe.design.Design` whose rows sum to 1. Proportions pass through
``Design.coded()`` unchanged; analysis uses Scheffé blending models (see
:func:`doe.analysis.model.build_model_matrix`).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import Factor, FactorSet, MixtureFactor

#: Proportions closer than this are treated as the same blend when deduplicating.
_DEDUP_DECIMALS = 9


def _validate_mixture(factors: Sequence[Factor]) -> FactorSet:
    """Build the all-mixture ``FactorSet`` (its constructor enforces the mixture rules)."""
    fs = FactorSet(factors)
    if not fs.is_mixture:
        non_mixture = [f.name for f in fs if not isinstance(f, MixtureFactor)]
        raise ValueError(
            "mixture designs require MixtureFactor components only; got non-mixture "
            f"factor(s): {non_mixture}"
        )
    return fs


def _require_unconstrained(fs: FactorSet, generator: str) -> None:
    """Lattice/centroid recipes cover the *full* simplex; bounds need extreme_vertices."""
    constrained = [
        f.name for f in fs if isinstance(f, MixtureFactor) and (f.low > 0.0 or f.high < 1.0)
    ]
    if constrained:
        raise ValueError(
            f"{generator} requires unconstrained components (low=0, high=1); "
            f"constrained component(s) {constrained} -- use extreme_vertices instead"
        )


def _to_design(
    points: np.ndarray,
    fs: FactorSet,
    *,
    name: str,
    meta: dict[str, object],
    point_types: tuple[str, ...],
) -> Design:
    runs = pd.DataFrame(points, columns=fs.names)
    return Design(runs, fs, name=name, meta=meta, point_types=point_types)


def _classify_blend(row: np.ndarray, n_components: int) -> str:
    """Point-type tag from the support (nonzero components) of a blend."""
    nonzero = row > 1e-12
    support = int(nonzero.sum())
    if support == 1:
        return "vertex"
    if support == n_components:
        return "centroid" if np.allclose(row, row[0]) else "interior"
    if support == 2 and np.allclose(row[nonzero], 0.5):
        return "edge-centroid"
    return "boundary"


def simplex_lattice(factors: Sequence[Factor], *, degree: int) -> Design:
    """The ``{k, m}`` simplex-lattice design: proportions on a ``1/m`` grid summing to 1.

    Every composition of ``degree`` parts among the ``k`` components, divided by ``degree``
    -- ``C(k + m - 1, m)`` runs whose proportions take values in ``{0, 1/m, ..., 1}``. The
    classic support for fitting a Scheffé polynomial of the same degree (``{3, 2}`` gives
    the six textbook points: three pure blends and three binary 50/50s). Requires
    unconstrained components; bound-constrained regions go to :func:`extreme_vertices`.
    """
    fs = _validate_mixture(factors)
    _require_unconstrained(fs, "simplex_lattice")
    if degree < 1:
        raise ValueError("degree must be a positive integer")

    k = len(fs)
    # compositions of `degree` into k non-negative parts, via stars-and-bars positions
    points = []
    for dividers in itertools.combinations(range(degree + k - 1), k - 1):
        bounds = (-1, *dividers, degree + k - 1)
        counts = [bounds[i + 1] - bounds[i] - 1 for i in range(k)]
        points.append([c / degree for c in counts])
    arr = np.asarray(points, dtype=float)
    point_types = tuple(_classify_blend(row, k) for row in arr)
    return _to_design(
        arr,
        fs,
        name=f"simplex_lattice_{k}_{degree}",
        meta={"generator": "simplex_lattice", "degree": degree},
        point_types=point_types,
    )


def simplex_centroid(factors: Sequence[Factor]) -> Design:
    """The simplex-centroid design: the ``2^k - 1`` centroids of every component subset.

    One run per non-empty subset ``S`` of components, blending the members of ``S`` equally
    (``1/|S|`` each): the ``k`` pure blends, the binary 50/50s, ..., up to the overall
    centroid. Requires unconstrained components.
    """
    fs = _validate_mixture(factors)
    _require_unconstrained(fs, "simplex_centroid")

    k = len(fs)
    points = []
    tags = []
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            row = np.zeros(k)
            row[list(subset)] = 1.0 / size
            points.append(row)
            if size == 1:
                tags.append("vertex")
            elif size == k:
                tags.append("centroid")
            elif size == 2:
                tags.append("edge-centroid")
            else:
                tags.append("face-centroid")
    return _to_design(
        np.asarray(points),
        fs,
        name=f"simplex_centroid_{k}",
        meta={"generator": "simplex_centroid"},
        point_types=tuple(tags),
    )


def _vertex_candidates(fs: FactorSet) -> np.ndarray:
    """XVERT-style vertex enumeration of the bound-constrained simplex.

    For every choice of a slack component, set each of the other ``k - 1`` components to its
    low or high bound and let the slack component absorb the remainder; keep the point when
    that remainder lies within the slack component's own bounds. Deduplicated. This is the
    McLean-Anderson (1966) extreme-vertices construction.
    """
    lows = np.array([f.low for f in fs if isinstance(f, MixtureFactor)])
    highs = np.array([f.high for f in fs if isinstance(f, MixtureFactor)])
    k = len(fs)

    vertices = []
    for slack in range(k):
        others = [j for j in range(k) if j != slack]
        for bits in itertools.product((0, 1), repeat=k - 1):
            row = np.empty(k)
            for j, bit in zip(others, bits, strict=True):
                row[j] = highs[j] if bit else lows[j]
            remainder = 1.0 - row[others].sum()
            if lows[slack] - 1e-12 <= remainder <= highs[slack] + 1e-12:
                row[slack] = remainder
                vertices.append(np.clip(row, lows, highs))
    if not vertices:  # pragma: no cover - FactorSet feasibility check makes this unreachable
        raise ValueError("no feasible vertices for the given component bounds")
    unique = np.unique(np.round(np.asarray(vertices), _DEDUP_DECIMALS), axis=0)
    return unique


def extreme_vertices(factors: Sequence[Factor], *, include_centroid: bool = True) -> Design:
    """Extreme-vertices design for a bound-constrained mixture region.

    Enumerates the vertices of the simplex clipped by the components' ``low``/``high``
    bounds (McLean-Anderson XVERT construction) and, with ``include_centroid`` (default),
    appends the region's overall centroid (the mean of the vertices) -- the workhorse for
    constrained formulation problems where lattice/centroid recipes do not apply.
    Rows are tagged ``"vertex"``/``"centroid"`` via ``point_types``, so a replicated
    centroid drives lack-of-fit exactly as center points do in box designs.
    """
    fs = _validate_mixture(factors)
    vertices = _vertex_candidates(fs)

    points = vertices
    tags: list[str] = ["vertex"] * len(vertices)
    if include_centroid:
        points = np.vstack([vertices, vertices.mean(axis=0)])
        tags.append("centroid")
    return _to_design(
        points,
        fs,
        name=f"extreme_vertices_{len(fs)}",
        meta={
            "generator": "extreme_vertices",
            "include_centroid": include_centroid,
            "n_vertices": int(len(vertices)),
        },
        point_types=tuple(tags),
    )


def mixture_candidates(factors: Sequence[Factor], *, resolution: int = 10) -> np.ndarray:
    """A discrete candidate set over the (possibly constrained) simplex.

    Returns an ``(n_candidates, k)`` array of proportions: the ``1/resolution`` lattice
    filtered to the feasible region, plus the extreme vertices and the region centroid.
    Shaped exactly like :func:`~doe.generators.optimal.candidate_grid` output, so it feeds
    :func:`~doe.generators.optimal.coordinate_exchange` / ``d_optimal(..., region=...)``
    directly -- and because the engine exchanges whole candidate rows, every design it
    builds satisfies the sum-to-1 constraint by construction.
    """
    fs = _validate_mixture(factors)
    if resolution < 1:
        raise ValueError("resolution must be a positive integer")

    lows = np.array([f.low for f in fs if isinstance(f, MixtureFactor)])
    highs = np.array([f.high for f in fs if isinstance(f, MixtureFactor)])
    k = len(fs)

    lattice = []
    for dividers in itertools.combinations(range(resolution + k - 1), k - 1):
        bounds = (-1, *dividers, resolution + k - 1)
        row = np.array([(bounds[i + 1] - bounds[i] - 1) / resolution for i in range(k)])
        if np.all(row >= lows - 1e-12) and np.all(row <= highs + 1e-12):
            lattice.append(row)

    vertices = _vertex_candidates(fs)
    stacked = np.vstack([*lattice, vertices, vertices.mean(axis=0)[None, :]])
    return np.asarray(np.unique(np.round(stacked, _DEDUP_DECIMALS), axis=0))
