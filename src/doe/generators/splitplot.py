"""Split-plot design generator (Phase 5b).

A split-plot design restricts randomization for *hard-to-change* factors: the hard-to-change
(whole-plot) factors are held constant across a group of runs -- a *whole plot* -- while the
easy-to-change (sub-plot) factors are varied within it. :func:`split_plot` builds one by
crossing a design on the whole-plot factors with a design on the sub-plot factors: each
whole-plot setting (optionally replicated) is one whole plot, and the full sub-plot design runs
inside it.

The result is a plain :class:`~doe.design.Design` carrying :attr:`~doe.design.Design.whole_plots`,
so :meth:`~doe.design.Design.randomize` is plot-aware (never splits a plot) and
:func:`~doe.analysis.fit.fit_gls` can recover the two-stratum error structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd

from ..design import Design
from ..factors import Factor, FactorSet, MixtureFactor
from .factorial import _generator_spec, full_factorial

ComponentDesign = Literal["full"] | Design


def _component_design(spec: ComponentDesign, subset: list[Factor], role: str) -> Design:
    """Resolve a whole-plot / sub-plot component design from a spec or a ready-made design."""
    if isinstance(spec, Design):
        want = {f.name for f in subset}
        got = set(spec.factors.names)
        if got != want:
            raise ValueError(
                f"the supplied {role} design is defined on factors {sorted(got)}, but the "
                f"{role} factors are {sorted(want)}; it must cover exactly those factors"
            )
        return spec
    if spec == "full":
        return full_factorial(subset)
    raise ValueError(f"{role}_design must be 'full' or a Design, got {spec!r}")


def split_plot(
    factors: Sequence[Factor],
    *,
    whole_plot_design: ComponentDesign = "full",
    sub_plot_design: ComponentDesign = "full",
    n_whole_plot_reps: int = 1,
    seed: int | None = None,
) -> Design:
    """Generate a split-plot design for hard-to-change (whole-plot) factors.

    Factors flagged :attr:`~doe.factors.ContinuousFactor.hard_to_change` form the whole-plot
    stratum; the rest form the sub-plot stratum. A design on the whole-plot factors is crossed
    with a design on the sub-plot factors: **each whole-plot setting × replicate is one whole
    plot**, and the whole sub-plot design is run inside it. A split-plot CCD or fractional
    factorial therefore falls out by composition -- pass a ready-made :class:`~doe.design.Design`
    on the corresponding factors as the component instead of ``"full"``.

    Args:
        factors: the full factor set; must contain at least one hard-to-change factor and at
            least one easy-to-change factor (otherwise it is an ordinary design).
        whole_plot_design: ``"full"`` (a full factorial on the whole-plot factors, built
            internally) or a :class:`~doe.design.Design` defined on exactly the whole-plot factors.
        sub_plot_design: ``"full"`` or a :class:`~doe.design.Design` on exactly the sub-plot
            factors, run inside every whole plot.
        n_whole_plot_reps: how many times to replicate the whole-plot design; each replicate of a
            whole-plot setting is a *distinct* whole plot (a physical re-setup).
        seed: when given, the design is returned already plot-aware-randomized via
            :meth:`~doe.design.Design.randomize` (the standard-order design is what you get with
            ``seed=None`` -- randomize it yourself for execution order).

    Returns:
        A :class:`~doe.design.Design` whose :attr:`~doe.design.Design.whole_plots` groups the runs,
        carrying the sub-plot design's ``point_types`` (repeated per plot) and recording the call
        in ``meta["generator"]``.

    Raises:
        ValueError: for mixture factors (split-plot × mixture is out of scope), fewer than one
            whole-plot or one sub-plot factor, ``n_whole_plot_reps < 1``, or a component design
            not defined on exactly its stratum's factors.
    """
    fs = FactorSet(factors)
    mixture = [f.name for f in fs if isinstance(f, MixtureFactor)]
    if mixture:
        raise ValueError(
            f"split_plot does not support mixture components {mixture}; split-plot pairs with "
            "the factorial/RSM families, not the simplex"
        )
    if n_whole_plot_reps < 1:
        raise ValueError("n_whole_plot_reps must be a positive integer")

    wp_factors = fs.whole_plot_factors
    sp_factors = fs.sub_plot_factors
    if not wp_factors or not sp_factors:
        raise ValueError(
            "split_plot needs at least one hard-to-change (whole-plot) factor and at least one "
            "easy-to-change (sub-plot) factor; with only one stratum this is an ordinary design "
            "-- use full_factorial / central_composite / d_optimal instead"
        )

    wp = _component_design(whole_plot_design, wp_factors, "whole_plot")
    sp = _component_design(sub_plot_design, sp_factors, "sub_plot")
    wp_names = [f.name for f in wp_factors]
    sp_names = [f.name for f in sp_factors]
    sp_point_types = sp.point_types

    rows: list[dict[str, object]] = []
    whole_plots: list[int] = []
    point_types: list[str] | None = [] if sp_point_types is not None else None
    plot_id = 0
    for _rep in range(n_whole_plot_reps):
        for w in range(wp.n_runs):
            wp_row = {name: wp.runs.iloc[w][name] for name in wp_names}
            for s in range(sp.n_runs):
                sp_row = {name: sp.runs.iloc[s][name] for name in sp_names}
                rows.append({**wp_row, **sp_row})
                whole_plots.append(plot_id)
                if point_types is not None and sp_point_types is not None:
                    point_types.append(sp_point_types[s])
            plot_id += 1

    runs = pd.DataFrame(rows, columns=fs.names)
    design = Design(
        runs,
        fs,
        name=f"split_plot_{len(wp_factors)}wp_{len(sp_factors)}sp",
        meta={
            "generator": _generator_spec(
                "split_plot",
                whole_plot_design="full" if whole_plot_design == "full" else "custom",
                sub_plot_design="full" if sub_plot_design == "full" else "custom",
                n_whole_plot_reps=n_whole_plot_reps,
                seed=seed,
            )
        },
        point_types=tuple(point_types) if point_types is not None else None,
        whole_plots=tuple(whole_plots),
    )
    return design.randomize(seed) if seed is not None else design
