.PHONY: install test lint format typecheck check docker clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=loom --cov-report=term-missing

lint:
	ruff check loom tests
	ruff format --check loom tests

format:
	ruff format loom tests
	ruff check --fix loom tests

typecheck:
	mypy

check: lint typecheck test

docker:
	docker build -t loom-lm .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info coverage.xml .coverage
