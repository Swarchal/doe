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


@dataclass(frozen=True)
class ContinuousFactor:
    """A continuous (numeric) factor varied between ``low`` and ``high``."""

    name: str
    low: float
    high: float
    units: str | None = None

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
        """JSON-ready representation, tagged with ``"type"`` for dispatch on load."""
        return {
            "type": "continuous",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuousFactor:
        return cls(
            name=str(data["name"]),
            low=float(data["low"]),
            high=float(data["high"]),
            units=data.get("units"),
        )


@dataclass(frozen=True)
class CategoricalFactor:
    """A categorical factor taking one of a fixed set of ``levels``.

    Unlike a continuous factor there is no natural midpoint or distance between levels, so
    categorical factors have no ``[-1, +1]`` coding of their own; they are turned into
    numeric contrast columns only at analysis time (see ``analysis.model._effect_code``).
    """

    name: str
    levels: tuple[object, ...]
    units: str | None = None

    def __post_init__(self) -> None:
        # a factor must vary to have an effect; a single level carries no information
        if len(self.levels) < 2:
            raise ValueError(f"factor {self.name!r}: needs at least 2 levels")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation; levels are stored as natural values, not encoded."""
        return {
            "type": "categorical",
            "name": self.name,
            "levels": list(self.levels),
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CategoricalFactor:
        return cls(
            name=str(data["name"]),
            levels=tuple(data["levels"]),
            units=data.get("units"),
        )


Factor = ContinuousFactor | CategoricalFactor


def factor_from_dict(data: Mapping[str, Any]) -> Factor:
    """Reconstruct a factor from its ``to_dict`` form, dispatching on ``"type"``."""
    kind = data.get("type")
    if kind == "continuous":
        return ContinuousFactor.from_dict(data)
    if kind == "categorical":
        return CategoricalFactor.from_dict(data)
    raise ValueError(f"unknown factor type {kind!r}; expected 'continuous' or 'categorical'")


class FactorSet:
    """An ordered collection of factors, addressable by name."""

    def __init__(self, factors: Sequence[Factor]):
        names = [f.name for f in factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique")
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ordered factor list; order fixes model-matrix column order."""
        return {"factors": [f.to_dict() for f in self._factors]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorSet:
        return cls([factor_from_dict(fd) for fd in data["factors"]])
