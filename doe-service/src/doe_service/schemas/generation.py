"""Request models for design-generation endpoints (Milestone 2).

One model per generator in ``docs/WEBSERVICE_API.md`` "Design generation" / "Optimal
designs": the shared ``factors: list[FactorSchema]`` plus that generator's own keyword
parameters, field-for-field with the table there.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from doe_service.schemas.design import DesignRequest
from doe_service.schemas.factors import FactorSchema


class FactorsRequest(BaseModel):
    """Shared base: every generation endpoint takes at least a factor list."""

    factors: list[FactorSchema]


class FullFactorialRequest(FactorsRequest):
    """``doe.full_factorial`` parameters."""

    levels: int | list[int] = 2


class FractionalFactorialRequest(FactorsRequest):
    """``doe.fractional_factorial`` parameters."""

    generators: list[str]


class PlackettBurmanRequest(FactorsRequest):
    """``doe.plackett_burman`` takes no parameters beyond ``factors``."""


class DefinitiveScreeningRequest(FactorsRequest):
    """``doe.definitive_screening`` parameters."""

    extra_center_runs: int = 0
    fake_factors: int | None = None


class CentralCompositeRequest(FactorsRequest):
    """``doe.central_composite`` parameters."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "factors": [
                        {
                            "type": "continuous", "name": "temp",
                            "low": 20, "high": 80, "units": "C",
                        },
                        {"type": "continuous", "name": "time", "low": 0, "high": 10},
                    ],
                    "alpha": "rotatable",
                    "center": 5,
                }
            ]
        }
    )

    alpha: Literal["faced", "rotatable", "orthogonal"] | float = "faced"
    center: int = 4
    fraction: list[str] | None = None


class BoxBehnkenRequest(FactorsRequest):
    """``doe.box_behnken`` parameters."""

    center: int = 3


class SpaceFillingRequest(FactorsRequest):
    """Dispatches to ``doe.latin_hypercube`` / ``doe.sobol`` / ``doe.halton`` on ``sampler``."""

    sampler: Literal["lhs", "sobol", "halton"]
    n_runs: int
    criterion: Literal["maximin", "correlation"] | None = "maximin"
    scramble: bool = True
    seed: int | None = None


class SimplexLatticeRequest(FactorsRequest):
    """``doe.simplex_lattice`` parameters."""

    degree: int


class SimplexCentroidRequest(FactorsRequest):
    """``doe.simplex_centroid`` takes no parameters beyond ``factors``."""


class ExtremeVerticesRequest(FactorsRequest):
    """``doe.extreme_vertices`` parameters."""

    include_centroid: bool = True


class OptimalRequest(FactorsRequest):
    """``doe.coordinate_exchange`` parameters (``docs/WEBSERVICE_API.md`` "Optimal designs").

    ``region`` is coded ``(m, k)`` candidate rows; ``None`` defaults to
    ``candidate_grid``/``mixture_candidates``.
    """

    n_runs: int
    model: Literal["linear", "quadratic"] = "quadratic"
    criterion: Literal["D", "I"] = "D"
    n_restarts: int = 20
    max_iter: int = 100
    seed: int | None = None
    region: list[list[float]] | None = None


class AugmentRequest(DesignRequest):
    """``doe.augment`` parameters — the posted design's rows are held fixed."""

    n_runs: int
    model: Literal["linear", "quadratic"] = "quadratic"
    criterion: Literal["D", "I"] = "D"
    n_restarts: int = 20
    max_iter: int = 100
    seed: int | None = None
    region: list[list[float]] | None = None


class CandidatesRequest(FactorsRequest):
    """``doe.candidate_grid`` (``levels``) / ``doe.mixture_candidates`` (``resolution``).

    Dispatch is on the factor set (all-mixture -> ``mixture_candidates``); only the
    parameter relevant to the resolved generator is used.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "factors": [
                        {"type": "continuous", "name": "temp", "low": 20, "high": 80},
                        {"type": "continuous", "name": "time", "low": 0, "high": 10},
                    ],
                    "levels": 3,
                }
            ]
        }
    )

    levels: int | None = None
    resolution: int | None = None
