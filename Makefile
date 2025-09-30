# Pandas TA Classic - Makefile
# ============================
# Quick reference for common development tasks
# Python version: the latest stable plus the prior 4 versions (dynamically managed via CI/CD)

# Package manager detection (prefer uv if available, fallback to pip)
PIP := $(shell command -v uv pip 2> /dev/null || echo pip)

.PHONY: all help clean install install-dev install-all test test-ext test-metrics test-strats test-ta test-utils docs validate

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
	@echo "  make test-ta          Run indicator tests"
	@echo "  make test-ext         Run extended indicator tests"
	@echo "  make test-utils       Run utility tests"
	@echo "  make test-metrics     Run metrics tests"
	@echo "  make test-strats      Run strategy tests"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build Sphinx documentation"
	@echo "  make docs-serve       Build and serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove Python cache files"
	@echo "  make validate         Validate package structure"
	@echo "  make lint             Run code quality checks"
	@echo "  make format           Format code with black"
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
	python -m unittest discover -s tests -p "test_ext_indicator_*.py" -v

test-metrics:
	python -m unittest tests.test_utils_metrics -v

test-strats:
	python -m unittest tests.test_strategy -v

test-ta:
	python -m unittest discover -s tests -p "test_indicator_*.py" -v

test-utils:
	python -m unittest tests.test_utils -v

# Documentation targets
docs:
	@echo "Building Sphinx documentation..."
	cd docs && make html
	@echo "Documentation built: docs/_build/html/index.html"

docs-serve: docs
	@echo "Starting local documentation server..."
	@echo "Open http://localhost:8000 in your browser"
	cd docs/_build/html && python -m http.server 8000

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
	find . -type f -name '*.pyc' -o -type d -name '__pycache__'

validate:
	@echo "Validating package structure..."
	python validate_structure.py

lint:
	@echo "Running flake8..."
	flake8 pandas_ta_classic --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 pandas_ta_classic --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=pandas_ta_classic/__init__.py

format:
	@echo "Formatting code with black..."
	black pandas_ta_classic/
	@echo "Checking import order with isort..."
	isort pandas_ta_classic/