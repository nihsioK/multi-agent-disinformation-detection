.PHONY: install dev test lint up down

install:
	uv sync
	uv run pre-commit install

dev:
	uv run uvicorn app.server:app --reload

test:
	uv run pytest -q

lint:
	uv run pre-commit run --all-files

up:
	docker compose up --build

down:
	docker compose down
