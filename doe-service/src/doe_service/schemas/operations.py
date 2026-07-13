"""Request models for design-operation endpoints (Milestone 2).

Pure transformations over a posted design document, per ``docs/WEBSERVICE_API.md``
"Design operations".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from doe_service.schemas.design import DesignRequest


class ValidateRequest(BaseModel):
    """``design`` is a *raw* mapping, not a validated :class:`DesignDocument`.

    ``/validate`` must accept a structurally invalid document and report it as 200
    payload (``{valid: false, errors: [...]}}``), not a Pydantic-shape 422 -- the one
    endpoint where an invalid design is data, not an error.
    """

    design: dict[str, Any]
    check_ranges: bool = False


class RandomizeRequest(DesignRequest):
    """``Design.randomize`` parameters."""

    seed: int | None = None


class ReplicateRequest(DesignRequest):
    """``Design.replicate`` parameters."""

    n: int
    each: bool = False


class ProjectRequest(DesignRequest):
    """``Design.project`` parameters — the post-screening "keep the survivors" step."""

    factors: list[str]


class ResponsesRequest(DesignRequest):
    """``Design.with_responses`` parameters — values aligned to run order."""

    responses: dict[str, list[float]]
