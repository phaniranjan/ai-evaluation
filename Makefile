.PHONY: help install install-dev format lint test test-dynamic test-static clean

help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dependencies + dev tools"
	@echo "  make format         Format code (black + isort)"
	@echo "  make lint           Run linters (flake8, pylint, mypy)"
	@echo "  make test           Run all tests"
	@echo "  make test-dynamic   Run only dynamic LLM-based tests"
	@echo "  make test-static    Run only static data tests"
	@echo "  make clean          Clean up cache and build files"
	@echo "  make pre-commit-install  Install pre-commit hooks"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	pylint src/ tests/ conftest.py
	mypy --ignore-missing-imports .

test:
	pytest -v

test-dynamic:
	pytest -v -m dynamic tests/dynamic/

test-static:
	pytest -v -m static tests/static/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .pylint.d
	rm -f test_execution.log

pre-commit-install:
	pre-commit install
	pre-commit run --all-files
