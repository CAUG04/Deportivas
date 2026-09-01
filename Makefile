.PHONY: help install lint format typecheck test test-cov check db-up db-down migrate

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Crea el entorno con las versiones exactas del lockfile
	uv sync --all-extras --frozen
	uv run pre-commit install

lint:  ## Ruff sin modificar ficheros
	uv run ruff check src tests
	uv run ruff format --check src tests

format:  ## Ruff con autofix
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:  ## mypy strict
	uv run mypy

test:  ## Tests
	uv run pytest

test-cov:  ## Tests con cobertura (el motor de backtest exige cobertura alta)
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Todo lo que corre CI

db-up:  ## Postgres local para desarrollo
	docker compose up -d postgres adminer

db-down:
	docker compose down

migrate:  ## Aplica migraciones Alembic sobre Postgres local
	uv run alembic upgrade head
