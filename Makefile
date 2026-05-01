.PHONY: format lint type test check

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

type:
	uv run mypy main.py tests/

test:
	uv run pytest -v

check: format lint type test
