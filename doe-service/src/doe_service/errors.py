"""Error envelope and exception handlers.

Implemented in Milestone 1 (``docs/WEBSERVICE_BUILD.md`` §1.3): the
``{"error": {"code", "message", "errors"}}`` envelope and the exception → status
mapping from ``docs/WEBSERVICE_API.md`` "Errors".
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from doe import ValidationError
from doe_service.limits import LimitExceeded

#: The spec's error codes, in the order of their table.
ERROR_CODES = ("validation_error", "infeasible", "limit_exceeded", "malformed", "internal")


class Infeasible(Exception):
    """Marks a library ``ValueError`` as a 422 ``infeasible`` response.

    Raised only by ``doe_service.convert.call_library`` -- there is deliberately no
    global ``ValueError`` handler, so a service bug (a bare ``ValueError`` raised
    somewhere in router code, not from a library call) still surfaces as 500
    ``internal`` rather than being mistaken for a domain-infeasible input.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ErrorBody(BaseModel):
    """``{code, message, errors}`` — the error object nested inside the envelope."""

    code: str
    message: str
    errors: list[str] = Field(default_factory=list)


class ErrorEnvelope(BaseModel):
    """``{"error": {...}}`` — the one envelope shape for every non-2xx response."""

    error: ErrorBody


def not_implemented() -> HTTPException:
    """The stub response for routes whose milestone has not landed yet."""
    return HTTPException(status_code=501, detail="not implemented; see docs/WEBSERVICE_BUILD.md")


def error_envelope(code: str, message: str, errors: list[str] | None = None) -> dict[str, Any]:
    """Build the ``{"error": {"code", "message", "errors"}}`` envelope body.

    Public (unlike the handlers below) so the Milestone 6 body-size middleware
    (``main.py``) can return the same envelope for its 413 response without going
    through the exception-handler machinery, which never sees a request whose body
    exceeds ``limits.max_body_bytes`` -- that check runs before FastAPI parses anything.
    """
    body = ErrorBody(code=code, message=message, errors=errors or [])
    return {"error": body.model_dump()}


def _format_pydantic_error(problem: dict[str, Any]) -> str:
    """Flatten one ``RequestValidationError.errors()`` entry into a readable string."""
    loc = ".".join(str(part) for part in problem.get("loc", ()))
    msg = problem.get("msg", "invalid")
    return f"{loc}: {msg}" if loc else str(msg)


async def _handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = exc.errors()
    # FastAPI reports a malformed JSON body as a RequestValidationError entry tagged
    # "json_invalid" (fastapi.routing) rather than raising json.JSONDecodeError itself.
    if any(problem.get("type") == "json_invalid" for problem in problems):
        return JSONResponse(
            status_code=400,
            content=error_envelope("malformed", "request body is not valid JSON"),
        )
    errors = [_format_pydantic_error(problem) for problem in problems]
    return JSONResponse(
        status_code=422,
        content=error_envelope("validation_error", "request failed validation", errors),
    )


async def _handle_doe_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ValidationError)
    return JSONResponse(
        status_code=422,
        content=error_envelope("validation_error", "design document is invalid", list(exc.errors)),
    )


async def _handle_infeasible(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, Infeasible)
    return JSONResponse(status_code=422, content=error_envelope("infeasible", exc.message))


async def _handle_limit_exceeded(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, LimitExceeded)
    return JSONResponse(status_code=422, content=error_envelope("limit_exceeded", str(exc)))


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_envelope("internal", "internal server error"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the envelope exception handlers on ``app``.

    Per ``docs/WEBSERVICE_API.md`` "Errors": ``RequestValidationError`` (Pydantic shape
    failures, and malformed-JSON bodies, distinguished by FastAPI's own ``json_invalid``
    tag) -> 422 ``validation_error``/400 ``malformed``; ``doe.ValidationError`` -> 422
    ``validation_error`` with its exhaustive ``.errors``; :class:`Infeasible` (raised by
    ``convert.call_library`` around library ``ValueError``s) -> 422 ``infeasible``;
    :class:`~doe_service.limits.LimitExceeded` -> 422 ``limit_exceeded``; anything else
    -> 500 ``internal`` with no library detail leaked. ``HTTPException`` (the
    :func:`not_implemented` 501 stubs) is left to FastAPI's own default handling, since
    it is looked up ahead of the catch-all ``Exception`` handler in Starlette's
    exception-middleware MRO walk.
    """
    app.add_exception_handler(RequestValidationError, _handle_request_validation_error)
    app.add_exception_handler(ValidationError, _handle_doe_validation_error)
    app.add_exception_handler(Infeasible, _handle_infeasible)
    app.add_exception_handler(LimitExceeded, _handle_limit_exceeded)
    app.add_exception_handler(Exception, _handle_unexpected)
