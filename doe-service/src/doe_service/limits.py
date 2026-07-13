"""Deployment-configurable parameter caps (``docs/WEBSERVICE_API.md`` "Limits").

The caps keep every endpoint synchronous in v1. Enforcement across routers lands in
Milestone 6 (``docs/WEBSERVICE_BUILD.md`` §6); the values are defined from the start so
implementations have one source for them.
"""

from dataclasses import dataclass


class LimitExceeded(Exception):
    """A request exceeded a cap; the M1 handlers map it to 422 ``limit_exceeded``."""


@dataclass(frozen=True)
class Limits:
    max_factors: int = 32
    max_runs: int = 10_000
    max_restarts: int = 20
    max_iter: int = 100
    max_region_rows: int = 100_000
    max_resolution: int = 200
    max_goals: int = 8
    max_body_bytes: int = 5 * 1024 * 1024


DEFAULT_LIMITS = Limits()
