"""Pydantic wire schemas, mirroring the shapes in ``docs/WEBSERVICE_API.md``.

Pydantic checks *shape*; ``doe.validate_design_dict`` (run by
``convert.design_from_document``) checks cross-field *consistency*. Fields land in
Milestone 1 (``docs/WEBSERVICE_BUILD.md`` §1.1).
"""
