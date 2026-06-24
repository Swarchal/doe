"""The :class:`Design` container -- the shared currency between generation and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import ContinuousFactor, FactorSet


@dataclass
class Design:
    """A set of experimental runs plus the factor metadata that produced them.

    ``runs`` holds the runs in *natural* units (one row per run, one column per factor).
    Responses are appended as additional columns once experiments are carried out.
    """

    runs: pd.DataFrame
    factors: FactorSet
    name: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    point_types: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        missing = set(self.factors.names) - set(self.runs.columns)
        if missing:
            raise ValueError(f"runs missing columns for factors: {sorted(missing)}")
        if self.point_types is not None and len(self.point_types) != len(self.runs):
            raise ValueError(
                f"point_types has {len(self.point_types)} entries but there are "
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
        return np.array(
            [i for i, t in enumerate(self.point_types) if t == "center"], dtype=int
        )

    def coded(self) -> pd.DataFrame:
        """Return the factor columns mapped to coded units.

        Continuous factors are mapped to ``[-1, +1]``; categorical factors are passed
        through unchanged (encoding into model columns is handled at analysis time).
        """
        out = {}
        for factor in self.factors:
            col = self.runs[factor.name]
            if isinstance(factor, ContinuousFactor):
                out[factor.name] = factor.code(col.to_numpy())
            else:
                out[factor.name] = col.to_numpy()
        return pd.DataFrame(out, index=self.runs.index)

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
        """
        if n < 1:
            raise ValueError("n must be a positive integer")
        base = np.arange(self.n_runs)
        idx = np.repeat(base, n) if each else np.tile(base, n)
        runs = self.runs.iloc[idx].reset_index(drop=True)
        point_types = (
            tuple(self.point_types[i] for i in idx) if self.point_types is not None else None
        )
        return Design(
            runs, self.factors, self.name, {**self.meta, "replicates": n}, point_types
        )

    def randomize(self, seed: int | None = None) -> Design:
        """Return a copy with the run order shuffled (records the original order).

        Randomizing the order in which runs are executed is a core DoE safeguard: it spreads
        any uncontrolled time-trend or lurking variable (drifting reagents, a warming room,
        operator fatigue) evenly across the factors instead of letting it correlate with -- and
        bias -- a particular effect. It is also what justifies treating the OLS residuals as
        independent. The original (standard-order) index is preserved as ``std_order`` so
        measured responses can be re-joined to the design after running in shuffled order.
        """
        rng = np.random.default_rng(seed)
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
            tuple(self.point_types[i] for i in order)
            if self.point_types is not None
            else None
        )
        return Design(
            shuffled, self.factors, self.name, {**self.meta, "randomized": True}, point_types
        )
