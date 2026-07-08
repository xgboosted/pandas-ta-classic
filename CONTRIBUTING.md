# Contributing to pandas-ta-classic

**pandas-ta-classic** is the community-maintained fork of the original pandas-ta library. Its goal is to provide a stable, actively-maintained, and comprehensive technical analysis library for pandas DataFrames.

We welcome contributions from the community! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python (latest stable plus prior 4 minor versions - see CI workflows for current supported versions)
- Git
- GitHub account

### Development Setup

1. **Fork and Clone**
 1. Go to [github.com/xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) and click **Fork**.
 2. Clone your fork:
 ```bash
 git clone https://github.com/your-username/pandas-ta-classic.git
 cd pandas-ta-classic
 ```

2. **Create Virtual Environment**
 
 Using `uv` (recommended):
 ```bash
 uv venv
 source .venv/bin/activate # On Windows: .venv\Scripts\activate
 ```
 
 Using `venv`:
 ```bash
 python -m venv .venv
 source .venv/bin/activate # On Windows: .venv\Scripts\activate
 ```

3. **Install Dependencies**
 
 Using `uv` (faster):
 ```bash
 # Install package in development mode with all dependencies
 uv pip install -e ".[all]"
 
 # Or install specific dependency groups as needed:
 uv pip install -e ".[dev]"          # Development dependencies
 uv pip install -e ".[test]"         # Testing: pytest, Hypothesis, coverage
 uv pip install -e ".[docs]"         # Documentation dependencies
 uv pip install -e ".[optional]"     # Optional runtime features (tqdm progress bars)
 uv pip install -e ".[oracle]"       # Oracle parity libs: TA-Lib + tulipy
 uv pip install -e ".[integration]"  # Backtesting integrations: backtesting, backtrader, vectorbt, yfinance
 uv pip install -e ".[performance]"  # Numba acceleration
 ```
 
 Using `pip`:
 ```bash
 # Install package in development mode with all dependencies
 pip install -e ".[all]"
 
 # Or install specific dependency groups as needed:
 pip install -e ".[dev]"          # Development dependencies
 pip install -e ".[test]"         # Testing: pytest, Hypothesis, coverage
 pip install -e ".[docs]"         # Documentation dependencies
 pip install -e ".[optional]"     # Optional runtime features (tqdm progress bars)
 pip install -e ".[oracle]"       # Oracle parity libs: TA-Lib + tulipy
 pip install -e ".[integration]"  # Backtesting integrations: backtesting, backtrader, vectorbt, yfinance
 pip install -e ".[performance]"  # Numba acceleration
 ```

## How to Contribute

### 1. Code Contributions

#### New Indicators
- Each indicator lives in its own file under the appropriate category directory (e.g., `pandas_ta_classic/momentum/new_indicator.py`). One public function per file matching the indicator name.
- Include comprehensive docstrings with examples
- Add type hints for all parameters and return types
- Include unit tests with edge cases
- **No need to manually update the Category dictionary** - indicators are automatically discovered from the filesystem

> **Note:** The library uses **dynamic category discovery** via `_build_category_dict()` in `_meta.py`. When you add a new indicator file to any category folder (e.g., `pandas_ta_classic/momentum/new_indicator.py`), it will automatically be detected and included in the `Category` dictionary. Just ensure your file is in the correct category folder!

> **Cross-package imports:** When one indicator imports another from a different category (e.g., ADX importing ATR), use the fully-qualified submodule path: `from pandas_ta_classic.volatility.atr import atr` — not `from pandas_ta_classic.volatility import atr`. The short form can return the module object instead of the function.

#### Bug Fixes
- Reference the issue number in your commit message
- Include regression tests to prevent reintroduction
- Update documentation if the fix changes behavior

#### Performance Improvements
- Include benchmarks showing improvement
- Ensure changes don't break existing functionality
- Document any new dependencies (e.g., Numba)

### 2. Testing

#### Running Tests
```bash
# Full test suite (primary — matches CI)
pytest tests/ -v

# Single test module (fastest feedback)
pytest tests/test_indicator_momentum.py -v

# Specific test
pytest tests/test_indicator_momentum.py::TestRSI::test_rsi -v

# Oracle tests (requires TA-Lib + tulipy: pip install -e ".[oracle]")
pytest tests/test_oracle_talib.py tests/test_oracle_tulipy.py -v

# Property-based tests (Hypothesis)
pytest tests/test_property_based.py -v

# Show Hypothesis input distribution statistics
pytest tests/test_property_based.py -v --hypothesis-show-statistics

# Use CI-optimized profile (more examples, longer deadline)
pytest tests/test_property_based.py -v --hypothesis-profile=ci

# Run with coverage
pytest --cov=pandas_ta_classic --cov-report=html
```

#### Writing Tests
- Use descriptive test names
- Test both normal and edge cases
- Include property-based tests for complex functions (see `tests/test_property_based.py` for examples)

#### Test Structure

Traditional unit test pattern:

```python
def test_indicator_name():
 """Test indicator_name with normal inputs."""
 data = pd.Series(range(100), dtype=float)
 result = indicator_name(data)
 assert result is not None
 assert len(result) == len(data)
```

Property-based test pattern (Hypothesis):

```python
import hypothesis.strategies as st
from hypothesis import assume, given, settings

@given(price_series(min_size=30, max_size=200), st.integers(min_value=2, max_value=20))
@settings(max_examples=100)
def test_my_indicator_invariant(s, length):
    """Output invariants must hold across all valid inputs."""
    assume(len(s) >= length + 2)
    result = ta.my_indicator(s, length=length)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    assert str(length) in result.name
```

See ``tests/test_property_based.py`` and ``docs/testing.rst`` for the full strategy catalog.

### 3. Documentation

#### Docstring Format
```python
from typing import Optional

def indicator_name(close: pd.Series, length: Optional[int] = None) -> pd.Series:
 """
 Brief description of the indicator.
 
 Longer description if needed, including mathematical formula
 or algorithm explanation.
 
 Args:
 close (pd.Series): Series of closing prices
 length (Optional[int]): Lookback period. Defaults to None (resolved internally, typically 20).
 
 Returns:
 pd.Series: Calculated indicator values
 
 Example:
 >>> import pandas as pd
 >>> import pandas_ta_classic as ta
 >>> data = pd.read_csv('data.csv')
 >>> result = ta.indicator_name(data['close'])
 """
```

#### README Updates
- Update indicator counts when adding new indicators
- Add examples for new features
- Keep installation instructions current

### 4. Code Style

#### Python Style Guide
- Follow PEP 8
- Use Black for formatting: `black pandas_ta_classic/` (line-length=150, skip-string-normalization)
- Use Ruff for linting: `ruff check pandas_ta_classic --select E9,F63,F7,F82`
- Use type hints for all functions
- Maximum line length: 150 characters (Black config)
- f-strings preferred over `.format()` or `%`-formatting

**Gate condition:** both `black --check --diff pandas_ta_classic/` and `ruff check pandas_ta_classic --select E9,F63,F7,F82` must return EXIT=0 before opening a PR. CI enforces this in the `code-quality` job.

#### Import Organization
```python
# Standard library
import math
from typing import Optional, Union

# Third-party
import numpy as np
import pandas as pd

# Local
from pandas_ta_classic.utils import verify_series
```

#### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### 5. Git Workflow

#### Branch Naming
- Features: `feat/<topic>`
- Bug fixes: `fix/<topic>`
- Documentation: `docs/<topic>`
- CI/workflow: `ci/<topic>`

One logical change per PR.

#### Commit Messages
```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`

#### Pull Request Process

1. **Create Feature Branch**
 ```bash
 git checkout -b feat/new-indicator
 ```

2. **Make Changes**
 - Write code with tests
 - Update documentation
 - Run tests and linting

3. **Commit Changes**
 ```bash
 # Stage specific files — avoid `git add .` to prevent accidentally including secrets or large files
 git add pandas_ta_classic/momentum/my_indicator.py tests/test_indicator_momentum.py
 git commit -m "feat(momentum): add new RSI variant"
 ```

4. **Push and Create PR**
 ```bash
 git push origin feat/new-indicator
 ```

5. **PR Requirements**
 - [ ] `black --check --diff pandas_ta_classic/` passes (EXIT=0)
 - [ ] `ruff check pandas_ta_classic --select E9,F63,F7,F82` passes (EXIT=0)
 - [ ] `pytest tests/ -v` passes
 - [ ] Documentation updated (docstrings + `docs/` if behavior changed)
 - [ ] Type hints included on all new function signatures
 - [ ] CHANGELOG.md `[Unreleased]` section updated
 - [ ] New indicator: entry added to `docs/indicators.rst`

## Version Management

Versions are managed automatically via [setuptools-scm](https://github.com/pypa/setuptools-scm) from git tags. **Never manually edit version strings.**

- Tagged commits produce a clean version (`0.4.0`)
- Untagged commits produce a dev version (`0.4.1.dev3` = 3 commits after `0.4.0`)
- Clone with full history — not a shallow clone — so setuptools-scm can find tags:
  ```bash
  git clone https://github.com/xgboosted/pandas-ta-classic.git
  ```

**Troubleshooting:** If `import pandas_ta_classic` shows version `0.0.0`, fetch missing tags:
```bash
git fetch --tags
pip install -e ".[dev]"  # reinstall to regenerate _version.py
```

### Creating a Release (maintainers only)

1. Rename `[Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`
2. Ensure CI is green: `pytest tests/ -v`
3. Create an **annotated** tag and push:
   ```bash
   git tag -a X.Y.Z -m "X.Y.Z"
   git push origin X.Y.Z
   ```
4. Draft a GitHub Release from that tag — publishing it triggers automated PyPI upload via CI.

## Roadmap

See [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues) for the full list. Current priority areas:

- Lazy loading
- Plugin system improvements

Contributions in these areas are especially welcome. Check the issues list for anything tagged `good first issue` or `help wanted`.

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn and contribute
- Focus on the technical merits of contributions

### Getting Help
- Check existing [Issues](https://github.com/xgboosted/pandas-ta-classic/issues)
- Ask questions in [Discussions](https://github.com/xgboosted/pandas-ta-classic/discussions)
- Review documentation and examples

### Reporting Issues
- Use issue templates when available
- Provide minimal reproducible examples
- Include system information (Python version, OS, etc.)
- Search existing issues before creating new ones

## Recognition

Contributors are recognized in:
- Git commit history
- Release notes for significant contributions

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [TA-Lib Documentation](https://ta-lib.org/doc/)
- [Technical Analysis Concepts](https://www.investopedia.com/technical-analysis-4689657)

## Questions?

Feel free to:
- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Contact maintainers for sensitive issues

Thank you for contributing to pandas-ta-classic! 