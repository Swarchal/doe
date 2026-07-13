"""The design document wire format and the shared design-carrying request/response.

Fields land in Milestone 1 (``docs/WEBSERVICE_BUILD.md`` §1.1).
"""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from doe_service.schemas.factors import FactorSchema


class DesignDocument(BaseModel):
    """The ``Design.to_dict()`` document (``docs/SERIALIZATION.md``).

    ``runs`` stays ``list[dict[str, Any]]`` — per-run key/level checking is
    ``validate_design_dict``'s job, already exhaustive; the schema does not duplicate it.

    ``whole_plots`` (split-plot designs only) is a per-run list of integer plot ids;
    like ``Design.to_dict`` it is emitted **only when set**, so a fully-randomized design
    serializes byte-for-byte as before and the existing generation contracts (which embed
    design documents and compare key sets exactly) are unaffected.
    """

    schema_version: str
    name: str | None = None
    factors: list[FactorSchema]
    runs: list[dict[str, Any]]
    point_types: list[str] | None = None
    whole_plots: list[int] | None = None
    meta: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _drop_unset_whole_plots(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if self.whole_plots is None:
            data.pop("whole_plots", None)
        return data


class DesignRequest(BaseModel):
    """Base for endpoints taking ``{design: DesignDocument, ...}``."""

    design: DesignDocument


class DesignResponse(BaseModel):
    """``{design, warnings}`` — every generator/operation response."""

    design: DesignDocument
    warnings: list[str] = []


class SearchReport(BaseModel):
    """The ``doe.OptimalDesign`` search diagnostics (``docs/WEBSERVICE_API.md``
    "Optimal designs"), carried at the top level of ``/optimal``/``/augment`` responses
    alongside ``design`` (not buried in ``meta``, which remains the durable serialized
    record) — it answers "did the search converge, how good is it".

    ``score`` is nullable: a degenerate (rank-deficient) search can score ``-inf``,
    which ``jsonable`` coerces to ``null`` per the "JSON has no NaN/Infinity" rule.
    """

    criterion: str
    score: float | None
    d_efficiency: float
    n_restarts: int
    converged: bool


class OptimalDesignResponse(BaseModel):
    """``{design, search, warnings}`` — the coordinate-exchange response shape."""

    design: DesignDocument
    search: SearchReport
    warnings: list[str] = []


class CandidatesResponse(BaseModel):
    """``{points, kind}`` — ``candidate_grid``/``mixture_candidates`` output."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "points": [
                        [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                        [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                    ],
                    "kind": "grid",
                }
            ]
        }
    )

    points: list[list[float]]
    kind: Literal["grid", "mixture"]


class ValidateResponse(BaseModel):
    """``{valid, errors}`` — ``/validate`` returns this with 200 either way."""

    valid: bool
    errors: list[str] = []
