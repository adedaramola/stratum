install:
	pip install -e ".[dev]"
	pre-commit install

install-local-embed:
	pip install -e ".[local-embed]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test-unit:
	pytest tests/unit/ -v --cov-fail-under=0

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test: test-unit

eval:
	pytest tests/e2e/test_ragas_eval.py -v --tb=short

ingest:
	stratum-ingest --source $(SOURCE)

docker-up:
	docker compose up -d && docker compose ps

docker-down:
	docker compose down -v

ci: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ .mypy_cache/ .ruff_cache/ reports/ .chroma/

.PHONY: install install-local-embed lint format typecheck test-unit test-integration \
        test-e2e test eval ingest docker-up docker-down ci clean
