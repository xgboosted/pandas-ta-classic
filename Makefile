# Pandas TA Classic - Makefile
# ============================
# Quick reference for common development tasks
# Python version: the latest stable plus the prior 4 versions (dynamically managed via CI/CD)

# Package manager detection (prefer uv if available, fallback to pip)
PIP := $(shell if command -v uv >/dev/null 2>&1; then echo "uv pip"; else echo "pip"; fi)

# Python: use the local venv if it exists, otherwise fall back to system python
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/python"; else echo "python3"; fi)

.PHONY: all help clean caches install install-dev install-all init test test-ext test-metrics test-strats test-ta test-utils test-all fixtures docs docs-serve lint format typecheck

# Default target
all: test

# Help command - show available targets
help:
	@echo "Pandas TA Classic - Development Commands"
	@echo "========================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install package in editable mode"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make install-all      Install with all optional dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-all         Run all tests against the committed fixtures"
	@echo "  make test-ta          Run indicator tests"
	@echo "  make test-ext         Run extended indicator tests"
	@echo "  make test-utils       Run utility tests"
	@echo "  make test-metrics     Run metrics tests"
	@echo "  make test-strats      Run strategy tests"
	@echo ""
	@echo "Fixtures:"
	@echo "  make fixtures         Regenerate expected_values.json + regression_snapshots.json"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build Sphinx documentation"
	@echo "  make docs-serve       Build and serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove Python cache files"
	@echo "  make lint             Run code quality checks"
	@echo "  make format           Format code with black"
	@echo "  make typecheck        Run mypy against the requires-python floor"
	@echo ""
	@echo "Package manager: $(PIP)"

# Installation targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[all]"

# Legacy target for backwards compatibility
init: install-dev
	@echo "Note: 'make init' is deprecated. Use 'make install-dev' instead."

# Testing targets
test: test-utils test-metrics test-ta test-ext test-strats

test-ext:
	$(PYTHON) -m unittest discover -s tests -p "test_ext_indicator_*.py" -v

test-metrics:
	$(PYTHON) -m unittest tests.test_utils_metrics -v

test-strats:
	$(PYTHON) -m unittest tests.test_strategy -v

test-ta:
	$(PYTHON) -m unittest discover -s tests -p "test_indicator_*.py" -v

test-utils:
	$(PYTHON) -m unittest tests.test_utils -v

# Regenerate JSON fixture files from TA-Lib oracle + native code.
# Deliberate step only — run after an INTENTIONAL algorithm change and review
# the resulting diff before committing.  Never wire this into a test target:
# a golden value the test run rewrites cannot detect a development error.
fixtures:
	@echo "Regenerating expected_values.json..."
	$(PYTHON) -m tests.fixtures.generate_fixtures
	@echo "Regenerating regression_snapshots.json..."
	$(PYTHON) -m tests.fixtures.generate_regression_snapshots
	@echo "Fixtures regenerated — review 'git diff tests/fixtures/' before committing."

# Run the full test suite against the committed fixtures
test-all:
	@echo "Running full test suite..."
	$(PYTHON) -m pytest tests/ -v

# Documentation targets
docs:
	@echo "Building Sphinx documentation..."
	cd docs && make html
	@echo "Documentation built: docs/_build/html/index.html"

docs-serve: docs
	@echo "Starting local documentation server..."
	@echo "Open http://localhost:8000 in your browser"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Maintenance targets
clean:
	@echo "Cleaning Python cache files..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .eggs 2>/dev/null || true
	@echo "Clean complete!"

caches:
	@echo "Finding Python cache files..."
	find . \( -type f -name '*.pyc' \) -o \( -type d -name '__pycache__' \)

lint:
	@echo "Running ruff..."
	ruff check .
	ruff check pandas_ta_classic --extend-select C901,E501 --exit-zero
	@echo "Checking black/ruff versions match .pre-commit-config.yaml..."
	$(PYTHON) tools/check_lint_versions.py

format:
	@echo "Formatting code with black..."
	black pandas_ta_classic/

# Target 3.12, not the 3.10 requires-python floor: numpy >= 2.5 stubs use PEP 695
# `type` statements that mypy cannot parse below 3.12. Runtime 3.10/3.11 support
# is covered by the testing-core matrix.
typecheck:
	@echo "Type checking against Python 3.12..."
	$(PYTHON) -m mypy --python-version 3.12
	@# core.pyi (the accessor stub for IDEs) shadows core.py during package
	@# discovery, so the accessor and strategy engine need their own pass.
	$(PYTHON) -m mypy --python-version 3.12 --no-warn-unused-configs pandas_ta_classic/core.py
