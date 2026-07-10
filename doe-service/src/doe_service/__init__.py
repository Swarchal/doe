"""Stateless HTTP API over the ``doe`` library.

The service is a thin FastAPI layer: every endpoint maps onto one public ``doe``
function, with the ``Design.to_dict()`` document as the wire format. The API contract
is specified in ``docs/WEBSERVICE_API.md``; the architecture decisions (why a separate
workspace package, why stateless) in ``docs/WEBSERVICE.md``.

The dependency points one way only: ``doe_service`` imports ``doe``, never the reverse.
"""

__version__ = "0.1.0"
