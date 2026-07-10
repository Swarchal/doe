# doe-service

A stateless HTTP API over the [`doe`](../README.md) library — a uv workspace member of
the `doe` repository, depending on the in-tree `doe` package.

- API contract: [`docs/WEBSERVICE_API.md`](../docs/WEBSERVICE_API.md)
- Architecture and rationale: [`docs/WEBSERVICE.md`](../docs/WEBSERVICE.md)

The dependency points one way only: `doe_service` imports `doe`, never the reverse —
`doe` stays on the scipy stack, and this package can be split into its own repository
later without surgery.

## Development

All commands from this directory:

```bash
uv run --extra dev pytest                              # tests
uv run --extra dev mypy                                # type-check (strict)
uv run --extra dev uvicorn --factory doe_service.main:create_app --reload   # dev server
```

Linting is repo-wide from the repository root: `uv run --extra dev ruff check .`.
