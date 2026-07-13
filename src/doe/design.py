"""The :class:`Design` container -- the shared currency between generation and analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._json import json_safe
from .factors import ContinuousFactor, FactorSet

#: Bumped when the serialized design shape changes incompatibly.
SCHEMA_VERSION = "1.0"


def _normalize_ids(ids: Sequence[int]) -> list[int]:
    """Relabel arbitrary integer group ids to 0, 1, 2, ... in first-appearance order."""
    mapping: dict[int, int] = {}
    out: list[int] = []
    for value in ids:
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return out


def _shuffle_within_groups(
    group_ids: Sequence[object], rng: np.random.Generator, *, shuffle_groups: bool
) -> np.ndarray:
    """Permutation of run indices that shuffles *within* each group, keeping groups contiguous.

    Groups are taken in first-appearance order. With ``shuffle_groups=True`` the group order is
    itself randomized (whole-plot order); with ``False`` it is preserved (block order). Runs
    inside every group are always shuffled. Never interleaves runs from different groups.
    """
    ids = list(group_ids)
    groups = list(dict.fromkeys(ids))
    if shuffle_groups:
        groups = [groups[i] for i in rng.permutation(len(groups))]
    members = {g: [i for i, v in enumerate(ids) if v == g] for g in groups}
    out: list[int] = []
    for g in groups:
        idx = members[g]
        out.extend(idx[i] for i in rng.permutation(len(idx)))
    return np.asarray(out, dtype=int)


def _draw_seed(seed: int | None) -> int:
    """Resolve ``seed`` to a concrete int, drawing one if unset.

    Recording the drawn seed (rather than leaving it ``None``) is what makes every
    randomized artifact regenerable from its serialized form. 32-bit keeps it within
    the safe-integer range of JSON/JavaScript consumers.
    """
    if seed is not None:
        return seed
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


@dataclass
class Design:
    """A set of experimental runs plus the factor metadata that produced them.

    ``runs`` holds the runs in *natural* units (one row per run, one column per factor).
    Responses are appended as additional columns once experiments are carried out.

    Examples:
        >>> import pandas as pd
        >>> from doe import ContinuousFactor, Design, FactorSet
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     ContinuousFactor("time", 5, 15),
        ... ])
        >>> design = Design(
        ...     pd.DataFrame({"temperature": [40, 80], "time": [5, 15]}),
        ...     factors,
        ...     name="two-run",
        ... )
        >>> design.n_runs
        2
        >>> design.coded().to_dict("list")
        {'temperature': [-1.0, 1.0], 'time': [-1.0, 1.0]}
    """

    runs: pd.DataFrame
    factors: FactorSet
    name: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    point_types: tuple[str, ...] | None = None
    #: Whole-plot assignment for a split-plot design: one integer plot id per run (``None`` for a
    #: fully-randomized design). Runs sharing an id are one whole plot and hold the hard-to-change
    #: factors constant. Carried through :meth:`replicate`/:meth:`randomize`/:meth:`project` like
    #: ``point_types``, and it makes :meth:`randomize` plot-aware.
    whole_plots: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        missing = set(self.factors.names) - set(self.runs.columns)
        if missing:
            raise ValueError(f"runs missing columns for factors: {sorted(missing)}")
        if self.point_types is not None and len(self.point_types) != len(self.runs):
            raise ValueError(
                f"point_types has {len(self.point_types)} entries but there are "
                f"{len(self.runs)} runs"
            )
        if self.whole_plots is not None and len(self.whole_plots) != len(self.runs):
            raise ValueError(
                f"whole_plots has {len(self.whole_plots)} entries but there are "
                f"{len(self.runs)} runs"
            )

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def n_center(self) -> int:
        """Number of center-point runs (0 when point types aren't tracked).

        Replicated center points serve two distinct DoE purposes: their spread gives a
        *pure-error* estimate independent of the model (the basis of the lack-of-fit test),
        and their mean vs. the factorial mean detects overall curvature in the response.
        """
        if self.point_types is None:
            return 0
        return sum(1 for t in self.point_types if t == "center")

    @property
    def center_indices(self) -> np.ndarray:
        """Positional indices of the center-point runs (empty when none/untracked)."""
        if self.point_types is None:
            return np.empty(0, dtype=int)
        return np.array([i for i, t in enumerate(self.point_types) if t == "center"], dtype=int)

    @property
    def n_whole_plots(self) -> int:
        """Number of distinct whole plots (0 when whole-plot structure isn't tracked)."""
        if self.whole_plots is None:
            return 0
        return len(set(self.whole_plots))

    def whole_plot_indices(self, plot: int) -> np.ndarray:
        """Positional indices of the runs in whole plot ``plot`` (empty when untracked)."""
        if self.whole_plots is None:
            return np.empty(0, dtype=int)
        return np.array([i for i, p in enumerate(self.whole_plots) if p == plot], dtype=int)

    def coded(self) -> pd.DataFrame:
        """Return the factor columns mapped to coded units.

        Continuous factors are mapped to ``[-1, +1]``; categorical factors are passed
        through unchanged (encoding into model columns is handled at analysis time).

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 60, 80]}), factors)
            >>> design.coded()["temperature"].tolist()
            [-1.0, 0.0, 1.0]
        """
        out = {}
        for factor in self.factors:
            col = self.runs[factor.name]
            if isinstance(factor, ContinuousFactor):
                out[factor.name] = factor.code(col.to_numpy())
            else:
                out[factor.name] = col.to_numpy()
        return pd.DataFrame(out, index=self.runs.index)

    def with_response(self, name: str, values: object) -> Design:
        """Return a copy with a measured response attached as a column of ``runs``.

        Keeping the response *on* the design is what stops it silently drifting out of
        alignment with the runs: the length is checked once here, and from then on the values
        ride along with their runs through :meth:`replicate` and :meth:`randomize` (which
        reindex every column together). Pass the resulting design straight to
        :func:`~doe.analysis.fit.fit_ols` by column name -- ``fit_ols(design, name)``.

        ``values`` must have exactly :attr:`n_runs` entries; a mismatch raises rather than
        producing a plausible-but-wrong fit. The original design is left unchanged (Designs are
        value objects), and ``name`` must not collide with a factor column.

        Attaching several responses means chaining calls (``design.with_response(...)
        .with_response(...)``); :meth:`with_responses` attaches them all in one call.

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 80]}), factors)
            >>> measured = design.with_response("yield", [12.5, 18.0])
            >>> list(measured.runs.columns)
            ['temperature', 'yield']
            >>> "yield" in design.runs.columns
            False
        """
        if name in self.factors.names:
            raise ValueError(f"response name {name!r} collides with a factor column")
        col = np.asarray(values)
        if col.ndim != 1:
            raise ValueError("response values must be one-dimensional")
        if col.shape[0] != self.n_runs:
            raise ValueError(
                f"response {name!r} has {col.shape[0]} values but there are {self.n_runs} runs"
            )
        runs = self.runs.copy()
        runs[name] = col
        return Design(
            runs, self.factors, self.name, dict(self.meta), self.point_types, self.whole_plots
        )

    def with_responses(self, **responses: object) -> Design:
        """Return a copy with several measured responses attached as columns of ``runs``.

        A convenience over chaining :meth:`with_response` once per column -- each keyword
        is attached in turn via :meth:`with_response`, so the same checks apply to every
        column (length == :attr:`n_runs`, one-dimensional, no collision with a factor name).
        Because responses attached earlier in the same call are legitimately present on the
        intermediate design by the time the next one is checked, one response name can never
        collide with another -- only with a factor column.

        Requires at least one keyword; called with none, it raises rather than silently
        returning an unchanged copy. Column names that aren't valid Python identifiers (so
        can't be passed as a keyword directly) still work via dict-unpacking, e.g.
        ``design.with_responses(**{"yield %": values})``.

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 80]}), factors)
            >>> measured = design.with_responses(yield_=[12.5, 18.0], impurity=[1.2, 0.8])
            >>> list(measured.runs.columns)
            ['temperature', 'yield_', 'impurity']
            >>> "yield_" in design.runs.columns
            False
        """
        if not responses:
            raise ValueError("no responses given")
        design = self
        for name, values in responses.items():
            design = design.with_response(name, values)
        return design

    def project(self, factors: Sequence[str]) -> Design:
        """Return a copy restricted to a subset of the factors, keeping every run.

        Drops the columns of the factors *not* named and narrows the :class:`FactorSet` to
        those that remain -- the runs themselves are untouched, so any response,
        ``std_order``, or other non-factor columns ride along aligned to their rows, and
        ``point_types`` carry through unchanged. This is the *project onto the survivors*
        step after a screen: once a :func:`~doe.plotting.half_normal_plot` has singled out
        the vital few factors, project the screening runs onto them and hand the result to
        :func:`~doe.generators.optimal.augment` to add curvature runs -- reusing the runs you
        already paid for rather than starting a fresh surface design.

        Because the dropped factors collapse, runs that differed only in a dropped column
        become repeats (a 2^(6-2) screen projected onto three factors becomes the 2^3 corners,
        each measured twice) -- exactly the replication a follow-up model benefits from.

        ``factors`` is given in the order you want the surviving factor columns to appear (it
        need not match the original order); each name must be a current factor and none may
        repeat. Note that projecting a mixture design is rejected downstream -- dropping a
        component breaks the sum-to-1 constraint (:class:`FactorSet` re-validates the subset).

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([
            ...     ContinuousFactor("temperature", 40, 80),
            ...     ContinuousFactor("time", 5, 15),
            ... ])
            >>> runs = pd.DataFrame({"temperature": [40, 80], "time": [5, 15]})
            >>> measured = Design(runs, factors).with_response("yield", [12.5, 18.0])
            >>> projected = measured.project(["temperature"])
            >>> projected.factors.names
            ['temperature']
            >>> list(projected.runs.columns)
            ['temperature', 'yield']
        """
        names = list(factors)
        if not names:
            raise ValueError("project needs at least one factor name")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate factor names in projection: {names}")
        unknown = [n for n in names if n not in self.factors.names]
        if unknown:
            raise ValueError(f"not factors of this design: {unknown}")
        keep = set(names)
        others = [c for c in self.runs.columns if c not in self.factors.names or c in keep]
        # surviving factor columns first (in requested order), then any non-factor columns
        ordered = names + [c for c in others if c not in keep]
        runs = self.runs.loc[:, ordered].copy()
        sub = FactorSet([self.factors[n] for n in names])
        return Design(runs, sub, self.name, dict(self.meta), self.point_types, self.whole_plots)

    def replicate(self, n: int, *, each: bool = False) -> Design:
        """Return a copy with every run replicated ``n`` times.

        With ``each=False`` (default) the whole design is repeated as ``n`` consecutive
        passes -- standard full replication, run order ``[all runs, all runs, ...]``.
        With ``each=True`` each run is repeated ``n`` times consecutively, so the
        replicates of one condition sit together -- convenient when the response array
        you measured groups its readings by condition.

        ``point_types`` are carried along (so center points stay labelled for pure-error /
        lack-of-fit) and ``meta["replicates"]`` records ``n``. Prefer this over stacking
        the ``runs`` frame by hand, which is easy to misalign with the response.

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 80]}), factors)
            >>> design.replicate(2).runs["temperature"].tolist()
            [40, 80, 40, 80]
            >>> design.replicate(2, each=True).runs["temperature"].tolist()
            [40, 40, 80, 80]
        """
        if n < 1:
            raise ValueError("n must be a positive integer")
        base = np.arange(self.n_runs)
        idx = np.repeat(base, n) if each else np.tile(base, n)
        runs = self.runs.iloc[idx].reset_index(drop=True)
        point_types = (
            tuple(self.point_types[i] for i in idx) if self.point_types is not None else None
        )
        whole_plots = self._replicate_whole_plots(n, each) if self.whole_plots is not None else None
        return Design(
            runs, self.factors, self.name, {**self.meta, "replicates": n}, point_types, whole_plots
        )

    def _replicate_whole_plots(self, n: int, each: bool) -> tuple[int, ...]:
        """Remap whole-plot ids under replication so each replicate pass is a *new* set of plots.

        Replicating a whole plot creates fresh plots (a physical re-setup of the hard-to-change
        factor), not more runs in the same plot -- so each pass offsets the (contiguity-normalized)
        plot ids by the plot count. With ``each=False`` a pass is one whole-design tile; with
        ``each=True`` it is the k-th consecutive copy of each run.
        """
        assert self.whole_plots is not None
        base = _normalize_ids(self.whole_plots)
        n_plots = max(base) + 1
        if each:
            return tuple(
                base[i] + rep * n_plots for i in range(self.n_runs) for rep in range(n)
            )
        return tuple(base[i] + rep * n_plots for rep in range(n) for i in range(self.n_runs))

    def randomize(self, seed: int | None = None, *, within: str | None = None) -> Design:
        """Return a copy with the run order shuffled (records the original order).

        Randomizing the order in which runs are executed is a core DoE safeguard: it spreads
        any uncontrolled time-trend or lurking variable (drifting reagents, a warming room,
        operator fatigue) evenly across the factors instead of letting it correlate with -- and
        bias -- a particular effect. It is also what justifies treating the OLS residuals as
        independent. The original (standard-order) index is preserved as ``std_order`` so
        measured responses can be re-joined to the design after running in shuffled order.

        Randomization respects restricted-randomization structure:

        * **Split-plot** (``whole_plots is not None``): the whole-plot order *and* the run order
          within each plot are both shuffled, but a plot is never split -- its runs stay
          contiguous. The whole-plot ids are relabelled to execution order (0, 1, 2, ...).
        * **Block-aware** (``within=`` a column name): the named column's groups are kept intact
          and *in their original order*; only the runs inside each group are shuffled. This is
          the run-order rule for blocked designs (the block column stays a contiguous, ordered
          nuisance stratum). ``within`` and split-plot structure are not combined.

        The seed actually used is recorded in ``meta["random_seed"]`` so the shuffle is
        reproducible (and serializable). When ``seed`` is ``None`` a concrete 32-bit seed is
        drawn and recorded rather than left unspecified, so every randomized design can be
        regenerated exactly.

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 60, 80]}), factors)
            >>> randomized = design.randomize(seed=1)
            >>> sorted(randomized.runs["std_order"].tolist())
            [0, 1, 2]
            >>> randomized.meta["random_seed"]
            1
        """
        seed = _draw_seed(seed)
        rng = np.random.default_rng(seed)
        relabel = False
        if within is not None:
            if within not in self.runs.columns:
                raise ValueError(
                    f"randomize(within={within!r}): no such column; "
                    f"available: {list(self.runs.columns)}"
                )
            order = _shuffle_within_groups(self.runs[within].to_list(), rng, shuffle_groups=False)
        elif self.whole_plots is not None:
            order = _shuffle_within_groups(list(self.whole_plots), rng, shuffle_groups=True)
            relabel = True
        else:
            order = rng.permutation(self.n_runs)

        shuffled = self.runs.iloc[order].reset_index(drop=True)
        base_order = (
            self.runs["std_order"].to_numpy()
            if "std_order" in self.runs
            else np.asarray(self.runs.index)
        )
        shuffled["std_order"] = base_order[order]
        shuffled = shuffled[["std_order", *[c for c in shuffled.columns if c != "std_order"]]]
        point_types = (
            tuple(self.point_types[i] for i in order) if self.point_types is not None else None
        )
        if self.whole_plots is not None:
            reordered = [self.whole_plots[i] for i in order]
            whole_plots: tuple[int, ...] | None = tuple(
                _normalize_ids(reordered) if relabel else reordered
            )
        else:
            whole_plots = None
        return Design(
            shuffled,
            self.factors,
            self.name,
            {**self.meta, "randomized": True, "random_seed": seed},
            point_types,
            whole_plots,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        Captures everything needed to reconstruct the design exactly: the ordered factor
        set, the full run table in *natural* units (including any ``std_order`` or response
        columns appended after randomization/experiments), the ``point_types`` that drive
        center-point / lack-of-fit logic, and ``meta`` (which carries the randomization seed).
        Coded values are intentionally *not* stored -- they are derived from the factor set.

        Examples:
            >>> import pandas as pd
            >>> from doe import ContinuousFactor, Design, FactorSet
            >>> factors = FactorSet([ContinuousFactor("temperature", 40, 80)])
            >>> design = Design(pd.DataFrame({"temperature": [40, 80]}), factors)
            >>> payload = design.to_dict()
            >>> payload["schema_version"]
            '1.0'
            >>> Design.from_dict(payload).coded()["temperature"].tolist()
            [-1.0, 1.0]
        """
        runs = [
            {key: json_safe(value) for key, value in record.items()}
            for record in self.runs.to_dict(orient="records")
        ]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            **self.factors.to_dict(),
            "runs": runs,
            "point_types": list(self.point_types) if self.point_types is not None else None,
            "meta": {key: json_safe(value) for key, value in self.meta.items()},
        }
        # emitted only when set, so fully-randomized designs serialize exactly as before
        if self.whole_plots is not None:
            payload["whole_plots"] = list(self.whole_plots)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Design:
        """Reconstruct a :class:`Design` from its :meth:`to_dict` form.

        The inverse of :meth:`to_dict`: ``Design.from_dict(d.to_dict())`` reproduces the run
        table, factor set, point types, and meta. Column order in ``runs`` is preserved from
        the serialized records.
        """
        factors = FactorSet.from_dict(data)
        records = list(data["runs"])
        columns = list(records[0].keys()) if records else factors.names
        runs = pd.DataFrame(records, columns=columns)
        point_types = data.get("point_types")
        whole_plots = data.get("whole_plots")
        return cls(
            runs,
            factors,
            name=str(data.get("name", "")),
            meta=dict(data.get("meta") or {}),
            point_types=tuple(point_types) if point_types is not None else None,
            whole_plots=tuple(int(p) for p in whole_plots) if whole_plots is not None else None,
        )
