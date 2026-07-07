"""The :class:`Design` container -- the shared currency between generation and analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .factors import ContinuousFactor, FactorSet

#: Bumped when the serialized design shape changes incompatibly.
SCHEMA_VERSION = "1.0"


def _jsonable(value: object) -> object:
    """Coerce numpy scalars to native Python types so the result is JSON-serializable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
                f"response {name!r} has {col.shape[0]} values but there are "
                f"{self.n_runs} runs"
            )
        runs = self.runs.copy()
        runs[name] = col
        return Design(runs, self.factors, self.name, dict(self.meta), self.point_types)

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
            shuffled,
            self.factors,
            self.name,
            {**self.meta, "randomized": True, "random_seed": seed},
            point_types,
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
            {key: _jsonable(value) for key, value in record.items()}
            for record in self.runs.to_dict(orient="records")
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            **self.factors.to_dict(),
            "runs": runs,
            "point_types": list(self.point_types) if self.point_types is not None else None,
            "meta": {key: _jsonable(value) for key, value in self.meta.items()},
        }

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
        return cls(
            runs,
            factors,
            name=str(data.get("name", "")),
            meta=dict(data.get("meta") or {}),
            point_types=tuple(point_types) if point_types is not None else None,
        )
