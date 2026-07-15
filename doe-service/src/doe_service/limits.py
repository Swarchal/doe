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

    # Auto-parallelism for the (expensive) optimal-design coordinate-exchange search. The
    # library exposes ``n_jobs`` to fan its independent restarts across worker processes, but
    # the HTTP layer never lets a *client* set it -- a single request must not be able to seize
    # the whole box. Instead the server decides: a search whose design reaches
    # ``optimal_parallel_min_runs`` runs is worth parallelising, and it may use up to
    # ``optimal_parallel_max_workers`` processes (``-1`` = all cores; the library further caps
    # workers at the restart count). Both default to *disabled* (``min_runs=0`` /
    # ``max_workers=1``), so the plain service stays single-process and byte-for-byte
    # reproducible; a deployment (e.g. doe-web) opts in by constructing ``Limits`` with these set.
    optimal_parallel_min_runs: int = 0
    optimal_parallel_max_workers: int = 1

    def optimal_n_jobs(self, n_runs: int) -> int:
        """The ``n_jobs`` to hand the coordinate-exchange search for a design of ``n_runs`` runs.

        Returns ``1`` (single-process) unless auto-parallelism is enabled *and* the design is at
        least ``optimal_parallel_min_runs`` runs; otherwise the configured worker cap.
        """
        if self.optimal_parallel_min_runs <= 0 or self.optimal_parallel_max_workers == 1:
            return 1
        if n_runs < self.optimal_parallel_min_runs:
            return 1
        return self.optimal_parallel_max_workers


DEFAULT_LIMITS = Limits()
