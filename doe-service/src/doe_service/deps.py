"""Shared FastAPI dependencies.

The one place the per-request :class:`~doe_service.limits.Limits` is pulled off
``app.state`` (stashed there by :func:`doe_service.main.create_app`). Every router
declares ``limits: Limits = Depends(app_limits)`` and threads it into the ``check_*``
helpers, so a deployment-configured cap (``create_app(limits=...)``) is honoured by every
size/budget guard -- not just the body-size middleware and the parallelism policy.
"""

from __future__ import annotations

from fastapi import Request

from doe_service.limits import DEFAULT_LIMITS, Limits


def app_limits(request: Request) -> Limits:
    """The deployment's :class:`Limits`, stashed on ``app.state`` by ``create_app``.

    Falls back to :data:`DEFAULT_LIMITS` if unset (e.g. an app built without the factory),
    so reading the caps never raises regardless of how the app was constructed.
    """
    return getattr(request.app.state, "limits", DEFAULT_LIMITS)
