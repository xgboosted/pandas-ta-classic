# Contributing to pandas-ta-classic

We welcome contributions to pandas-ta-classic! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python (latest stable plus prior 4 minor versions - see CI workflows for current supported versions)
- Git
- GitHub account

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/pandas-ta-classic.git
   cd pandas-ta-classic
   ```

2. **Create Virtual Environment**
   
   Using `uv` (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   
   Using `venv`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   
   Using `uv` (faster):
   ```bash
   # Install package in development mode with all dependencies
   uv pip install -e ".[all]"
   
   # Or install specific dependency groups as needed:
   uv pip install -e ".[dev]"        # Development dependencies
   uv pip install -e ".[test]"       # Testing dependencies only
   uv pip install -e ".[docs]"       # Documentation dependencies
   uv pip install -e ".[optional]"   # Optional runtime dependencies
   ```
   
   Using `pip`:
   ```bash
   # Install package in development mode with all dependencies
   pip install -e ".[all]"
   
   # Or install specific dependency groups as needed:
   pip install -e ".[dev]"        # Development dependencies
   pip install -e ".[test]"       # Testing dependencies only
   pip install -e ".[docs]"       # Documentation dependencies
   pip install -e ".[optional]"   # Optional runtime dependencies
   ```

## How to Contribute

### 1. Code Contributions

#### New Indicators
- Add indicators to the appropriate category module (momentum, overlap, trend, etc.)
- Include comprehensive docstrings with examples
- Add type hints for all parameters and return types
- Include unit tests with edge cases
- **No need to manually update the Category dictionary** - indicators are automatically discovered from the filesystem

> **Note:** The library uses **dynamic category discovery** via `_build_category_dict()` in `_meta.py`. When you add a new indicator file to any category folder (e.g., `pandas_ta_classic/momentum/new_indicator.py`), it will automatically be detected and included in the `Category` dictionary. Just ensure your file is in the correct category folder!

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
# Run all tests
pytest

# Run specific test file
pytest tests/test_indicator_momentum.py

# Run with coverage
pytest --cov=pandas_ta_classic --cov-report=html
```

#### Writing Tests
- Use descriptive test names
- Test both normal and edge cases
- Include property-based tests for complex functions
- Add performance tests for critical functions

#### Test Structure
```python
def test_indicator_name():
    """Test indicator_name with normal inputs."""
    # Arrange
    data = sample_data()
    
    # Act
    result = indicator_name(data)
    
    # Assert
    assert result is not None
    assert len(result) == len(data)
```

### 3. Documentation

#### Docstring Format
```python
def indicator_name(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Brief description of the indicator.
    
    Longer description if needed, including mathematical formula
    or algorithm explanation.
    
    Args:
        close (pd.Series): Series of closing prices
        length (int, optional): Lookback period. Defaults to 20.
    
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
- Use Black for code formatting: `black .`
- Use type hints for all functions
- Maximum line length: 88 characters

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
- Features: `feature/indicator-name` or `feature/description`
- Bug fixes: `fix/issue-number-description`
- Documentation: `docs/description`
- Performance: `perf/description`

#### Commit Messages
```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `perf`

#### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-indicator
   ```

2. **Make Changes**
   - Write code with tests
   - Update documentation
   - Run tests and linting

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(momentum): add new RSI variant"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/new-indicator
   ```

5. **PR Requirements**
   - [ ] Tests pass
   - [ ] Code coverage maintained/improved
   - [ ] Documentation updated
   - [ ] Type hints included
   - [ ] No linting errors

## Version Management

The project uses [setuptools-scm](https://github.com/pypa/setuptools-scm) for automatic version management based on git tags.

### How It Works

Version numbers are automatically determined from git tags and commit history:

- **Tagged releases**: Version matches the git tag exactly (e.g., `0.4.0`)
- **Development builds**: Version includes `.post` suffix with commit count (e.g., `0.3.15.post12`)

#### Version Scheme

We use the `post-release` version scheme:
```
0.3.15.post12  # 12 commits after tag 0.3.15
0.4.0          # Clean release from tag 0.4.0
```

This scheme keeps the base version from the most recent tag and appends a `.post` suffix with the number of commits since that tag.

### For Contributors

When working on the repository:

- **Never manually edit version strings** - versions are automatically determined from git tags
- Development builds show `.post` suffix (e.g., `0.3.15.post12` = 12 commits after 0.3.15 tag)
- Clone with full history (not shallow): `git clone https://github.com/xgboosted/pandas-ta-classic.git`
- Check current version:
  ```python
  import pandas_ta_classic as ta
  print(ta.version)  # Should show something like 0.3.15.post12
  ```

### For Maintainers

#### Creating a Release

1. **Update CHANGELOG.md**
   - Move items from "Unreleased" to new version section
   - Add release date
   - Follow [Keep a Changelog](https://keepachangelog.com/) format

2. **Ensure all tests pass**
   - Check CI/CD status
   - Run tests locally: `pytest`

3. **Create and push git tag**
   ```bash
   # Ensure you're on main branch
   git checkout main
   git pull
   
   # Create annotated tag
   git tag -a 0.4.0 -m "Release version 0.4.0"
   
   # Push tag to GitHub
   git push origin 0.4.0
   ```

4. **Create GitHub Release**
   - Go to repository's Releases page
   - Click "Draft a new release"
   - Select the tag you just created
   - Copy relevant CHANGELOG.md entries
   - Publish the release
   
5. **Automated Publishing**
   - CI/CD workflow automatically publishes to PyPI when GitHub Release is published
   - Version is automatically set to tag name (e.g., `0.4.0`)
   - No manual version updates needed in any files
   - Verify on PyPI that package appears with correct version

#### Version Resolution

The package resolves version in the following order:

1. `pandas_ta_classic/_version.py` (generated by setuptools-scm during installation)
2. Installed package metadata via `importlib.metadata`
3. Installed package metadata via `pkg_resources` (Python < 3.8)
4. Fallback to `"0.0.0"` if not installed

#### Configuration

Version management is configured in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm>=8.0", "wheel"]

[project]
dynamic = ["version"]

[tool.setuptools_scm]
write_to = "pandas_ta_classic/_version.py"
version_scheme = "post-release"
local_scheme = "no-local-version"
fallback_version = "0.0.0"
```

### CI/CD Considerations

#### GitHub Actions Shallow Clones

GitHub Actions by default performs shallow clones (only the latest commit), which prevents setuptools-scm from accessing git tags and history. All checkout actions in CI/CD workflows must use:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0  # Full history needed for setuptools-scm
```

Without `fetch-depth: 0`, setuptools-scm will fall back to `0.0.0`.

#### Fallback Version

If git is not available (e.g., building from a source tarball), the version falls back to `0.0.0`. This is a valid PEP 440 version that won't conflict with the post-release scheme.

### Troubleshooting

#### Problem: Installation fails with `Invalid version: '0.0.0.dev0.post1'`

**Cause**: 
- Git repository is shallow (missing tags/history)
- Fallback version uses `.dev0` suffix which conflicts with post-release scheme

**Solution**:
- In CI/CD: Add `fetch-depth: 0` to all checkout actions
- Locally: Ensure you have full git history and tags (`git fetch --unshallow --tags`)
- Update fallback version to `0.0.0` (without `.dev0`)

#### Problem: Version shows `0.0.0` instead of expected version

**Cause**: setuptools-scm cannot find git tags or history

**Solution**: 
- Ensure git tags exist: `git tag -l`
- Fetch tags if missing: `git fetch --tags`
- Check setuptools-scm is installed: `pip show setuptools-scm`
- Verify full git history: `git log --oneline` (should show all commits)

#### Problem: Development version not incrementing

**Cause**: Working from a clean tag without additional commits

**Solution**: This is expected behavior. Make commits after the tag to see `.post` suffix increment.

### Benefits

- **No manual version updates**: Version is always correct based on git tags
- **No version conflicts**: Single source of truth (git tags)
- **Development clarity**: `.post` suffix clearly indicates development builds
- **PEP 440 compliant**: All versions follow Python versioning standards
- **CI/CD friendly**: Works automatically in GitHub Actions with proper configuration

## Project Structure

```
pandas_ta_classic/
├── __init__.py
├── core.py              # Core functionality
├── custom.py            # Custom indicators
├── candles/            # Candlestick patterns
├── cycles/             # Cycle indicators  
├── momentum/           # Momentum indicators
├── overlap/            # Overlap studies
├── performance/        # Performance metrics
├── statistics/         # Statistical functions
├── trend/              # Trend indicators
├── utils/              # Utility functions
├── volatility/         # Volatility indicators
└── volume/             # Volume indicators
```

## Current Roadmap

See our [GitHub Issues](https://github.com/xgboosted/pandas-ta-classic/issues) for the current roadmap. Priority areas include:

### Phase 1: Foundation
- Type hints for all modules
- Standardized error handling
- Comprehensive test coverage

### Phase 2: Performance  
- Numba optimization
- Lazy loading
- Performance benchmarks

### Phase 3: API Enhancements
- Fluent API for chaining
- Plugin system improvements
- Custom metrics support

### Phase 4: Integrations
- vectorbt integration
- backtrader integration
- Enhanced TA-Lib support

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
- README contributors section (planned)

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

Thank you for contributing to pandas-ta-classic! 🚀