# fastapi-hello

[![CI](https://github.com/manoharkancharla/fastapi-hello/actions/workflows/ci.yml/badge.svg)](https://github.com/manoharkancharla/fastapi-hello/actions/workflows/ci.yml)

Minimal FastAPI service exploring async patterns and HTTP client mocking. Built as Week 1 of a structured AI Engineering pivot.

## Endpoints

- `GET /health` — liveness check, returns `{"status": "ok"}`.
- `GET /joke` — proxies a public joke API, returns `{setup, punchline}`. Returns `502` if the upstream is unreachable or errors.

## Stack

- Python 3.12, FastAPI, httpx (async), Pydantic v2
- pytest, respx (HTTP mocking), mypy (strict), ruff
- uv for dependency management, GitHub Actions for CI

## Local development

```bash
uv sync --dev
uv run uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the OpenAPI UI.

## Running checks

```bash
make check        # format + lint + type + test
make test         # tests only
```

## Notes on design

- `fetch_joke()` is extracted from the route handler so the upstream call is testable without hitting the real network.
- HTTP timeout is explicit (5s); both connection failures (`httpx.RequestError`) and non-200 responses are translated to `502 Bad Gateway`.
- `mypy --strict` is enabled from day one to catch type drift early.
