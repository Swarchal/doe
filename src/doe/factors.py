"""Factor definitions and coding between natural and coded units.

Designs are generated and analysed in *coded* units (continuous factors mapped to
``[-1, +1]``) but reported in *natural* units (the real values an experimenter sets).
The classes here own that translation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._json import json_safe


@dataclass(frozen=True)
class ContinuousFactor:
    """A continuous (numeric) factor varied between ``low`` and ``high``.

    Examples:
        >>> import numpy as np
        >>> temperature = ContinuousFactor("temperature", low=40, high=80, units="C")
        >>> temperature.center
        60.0
        >>> temperature.code(np.array([40, 60, 80])).tolist()
        [-1.0, 0.0, 1.0]
        >>> temperature.decode(np.array([-1, 0, 1])).tolist()
        [40.0, 60.0, 80.0]
    """

    name: str
    low: float
    high: float
    units: str | None = None
    #: When ``True`` the factor is *hard to change* -- it cannot be reset every run, so a
    #: split-plot design holds it constant within a whole plot (see
    #: :func:`doe.generators.splitplot.split_plot`). Ignored by the fully-randomized generators.
    hard_to_change: bool = False

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(
                f"factor {self.name!r}: high ({self.high}) must exceed low ({self.low})"
            )

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def code(self, values: np.ndarray) -> np.ndarray:
        """Map natural units to coded ``[-1, +1]`` units.

        Centering on the midpoint and scaling by the half-range is what makes effects
        comparable across factors measured on different scales (e.g. temperature in degrees
        vs. concentration in molar): a one-unit move in coded space is the full low->high
        swing for every factor. It also keeps the main-effect columns balanced about zero, so
        in a balanced design they are mutually orthogonal and the effect estimates do not
        confound one another.
        """
        return (np.asarray(values, dtype=float) - self.center) / self.half_range

    def decode(self, coded: np.ndarray) -> np.ndarray:
        """Map coded ``[-1, +1]`` units back to natural units."""
        return np.asarray(coded, dtype=float) * self.half_range + self.center

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation, tagged with ``"type"`` for dispatch on load.

        ``hard_to_change`` is emitted only when ``True`` so existing serialized designs stay
        byte-for-byte stable.
        """
        data: dict[str, Any] = {
            "type": "continuous",
            "name": self.name,
            "low": json_safe(self.low),
            "high": json_safe(self.high),
            "units": self.units,
        }
        if self.hard_to_change:
            data["hard_to_change"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuousFactor:
        return cls(
            name=str(data["name"]),
            low=float(data["low"]),
            high=float(data["high"]),
            units=data.get("units"),
            hard_to_change=bool(data.get("hard_to_change", False)),
        )


@dataclass(frozen=True)
class CategoricalFactor:
    """A categorical factor taking one of a fixed set of ``levels``.

    Unlike a continuous factor there is no natural midpoint or distance between levels, so
    categorical factors have no ``[-1, +1]`` coding of their own; they are turned into
    numeric contrast columns only at analysis time (see ``analysis.model._effect_code``).

    Examples:
        >>> catalyst = CategoricalFactor("catalyst", levels=("A", "B", "C"))
        >>> catalyst.levels
        ('A', 'B', 'C')
        >>> catalyst.to_dict()["type"]
        'categorical'
    """

    name: str
    levels: tuple[object, ...]
    units: str | None = None
    #: See :attr:`ContinuousFactor.hard_to_change` -- a hard-to-change categorical factor is a
    #: whole-plot factor in a split-plot design.
    hard_to_change: bool = False

    def __post_init__(self) -> None:
        # a factor must vary to have an effect; a single level carries no information
        if len(self.levels) < 2:
            raise ValueError(f"factor {self.name!r}: needs at least 2 levels")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation; levels are stored as natural values, not encoded.

        ``hard_to_change`` is emitted only when ``True`` (existing documents stay stable).
        """
        data: dict[str, Any] = {
            "type": "categorical",
            "name": self.name,
            "levels": [json_safe(level) for level in self.levels],
            "units": self.units,
        }
        if self.hard_to_change:
            data["hard_to_change"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CategoricalFactor:
        return cls(
            name=str(data["name"]),
            levels=tuple(data["levels"]),
            units=data.get("units"),
            hard_to_change=bool(data.get("hard_to_change", False)),
        )


@dataclass(frozen=True)
class MixtureFactor:
    """A mixture component: a proportion of the whole, in ``[low, high] ⊆ [0, 1]``.

    Mixture components are *proportions that sum to 1* across a run, so their design region
    is a simplex, not a box. Proportions are already dimensionless and bounded, so mixture
    columns are deliberately **not** rescaled to ``[-1, +1]``: :meth:`doe.Design.coded`
    passes them through unchanged, and Scheffé blending models (see
    :func:`doe.analysis.model.build_model_matrix`) are defined directly on the proportions.
    A ``FactorSet`` must be *all* mixture components or none (see :class:`FactorSet`).

    Examples:
        >>> a = MixtureFactor("A", low=0.1, high=0.8)
        >>> b = MixtureFactor("B", low=0.2, high=0.9)
        >>> factors = FactorSet([a, b])
        >>> factors.is_mixture
        True
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    units: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError(
                f"factor {self.name!r}: proportion bounds must satisfy "
                f"0 <= low < high <= 1, got [{self.low}, {self.high}]"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation, tagged with ``"type"`` for dispatch on load."""
        return {
            "type": "mixture",
            "name": self.name,
            "low": json_safe(self.low),
            "high": json_safe(self.high),
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MixtureFactor:
        return cls(
            name=str(data["name"]),
            low=float(data.get("low", 0.0)),
            high=float(data.get("high", 1.0)),
            units=data.get("units"),
        )


Factor = ContinuousFactor | CategoricalFactor | MixtureFactor


def factor_from_dict(data: Mapping[str, Any]) -> Factor:
    """Reconstruct a factor from its ``to_dict`` form, dispatching on ``"type"``."""
    kind = data.get("type")
    if kind == "continuous":
        return ContinuousFactor.from_dict(data)
    if kind == "categorical":
        return CategoricalFactor.from_dict(data)
    if kind == "mixture":
        return MixtureFactor.from_dict(data)
    raise ValueError(
        f"unknown factor type {kind!r}; expected 'continuous', 'categorical', or 'mixture'"
    )


class FactorSet:
    """An ordered collection of factors, addressable by name.

    The order is significant: generators, :class:`~doe.design.Design.coded`, and model
    matrices all use this order when producing columns.

    Examples:
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     CategoricalFactor("catalyst", ("A", "B")),
        ... ])
        >>> factors.names
        ['temperature', 'catalyst']
        >>> factors["temperature"].low
        40
        >>> [factor.name for factor in factors]
        ['temperature', 'catalyst']
    """

    def __init__(self, factors: Sequence[Factor]):
        names = [f.name for f in factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique")
        mixture = [f for f in factors if isinstance(f, MixtureFactor)]
        if mixture and len(mixture) != len(factors):
            # mixture-plus-process-variable designs are deferred (see docs/PHASE4.md):
            # proportions live on a simplex and Scheffé models have no intercept, so
            # mixing the two factor kinds would need a combined model form we don't have.
            other = sorted(f.name for f in factors if not isinstance(f, MixtureFactor))
            raise ValueError(
                f"mixture components cannot be combined with other factor types; "
                f"non-mixture factor(s): {other}"
            )
        if mixture:
            if len(mixture) < 2:
                raise ValueError("a mixture design needs at least 2 components")
            sum_low = sum(f.low for f in mixture)
            sum_high = sum(f.high for f in mixture)
            if not sum_low <= 1.0 <= sum_high:
                raise ValueError(
                    "mixture component bounds leave no feasible blend: need "
                    f"sum(low) <= 1 <= sum(high), got sum(low)={sum_low}, "
                    f"sum(high)={sum_high}"
                )
        self._factors: tuple[Factor, ...] = tuple(factors)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self._factors)

    def __len__(self) -> int:
        return len(self._factors)

    def __getitem__(self, key: int | str) -> Factor:
        if isinstance(key, str):
            for f in self._factors:
                if f.name == key:
                    return f
            raise KeyError(key)
        return self._factors[key]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self._factors]

    @property
    def is_mixture(self) -> bool:
        """``True`` if every factor is a :class:`MixtureFactor` (all-mixture designs only)."""
        return bool(self._factors) and all(isinstance(f, MixtureFactor) for f in self._factors)

    @property
    def whole_plot_factors(self) -> list[Factor]:
        """The hard-to-change factors -- the whole-plot stratum of a split-plot design."""
        return [f for f in self._factors if getattr(f, "hard_to_change", False)]

    @property
    def sub_plot_factors(self) -> list[Factor]:
        """The easy-to-change factors -- the sub-plot stratum (everything not hard-to-change)."""
        return [f for f in self._factors if not getattr(f, "hard_to_change", False)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ordered factor list; order fixes model-matrix column order."""
        return {"factors": [f.to_dict() for f in self._factors]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorSet:
        return cls([factor_from_dict(fd) for fd in data["factors"]])
