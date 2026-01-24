.PHONY: help test test-quick lint format check install-dev install-hooks clean

help:
	@echo "🛠️  The Briefing - Development Commands"
	@echo ""
	@echo "  make install-dev    Install development dependencies"
	@echo "  make install-hooks  Install pre-commit hooks"
	@echo "  make test          Run full test suite"
	@echo "  make test-quick    Run quick smoke tests"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make format        Format code with ruff"
	@echo "  make check         Run all checks (lint + test)"
	@echo "  make clean         Clean cache and build files"

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install pytest pytest-cov pytest-asyncio ruff mypy pre-commit

install-hooks:
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install

test:
	@echo "🧪 Running full test suite..."
	pytest tests/ -v --tb=short

test-quick:
	@echo "⚡ Running quick smoke tests..."
	pytest tests/test_basic_imports.py tests/test_run_functions.py -v

lint:
	@echo "🔍 Running linters..."
	@echo "→ Ruff check..."
	ruff check . --select F,E,W,N
	@echo "→ MyPy check..."
	mypy run.py --ignore-missing-imports --check-untyped-defs || true

format:
	@echo "✨ Formatting code..."
	ruff format .
	ruff check . --fix --select I

check: lint test-quick
	@echo "✅ All checks passed!"

clean:
	@echo "🧹 Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✨ Clean!"
