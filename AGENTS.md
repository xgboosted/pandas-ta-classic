# AGENTS.md

## Repository Context

- **Project:** pandas-ta-classic — Community-maintained Python 3 technical analysis library with 250+ indicators across 10 categories (Candles, Cycles, Momentum, Overlap, Performance, Statistics, Trend, Volatility, Volume, Math) plus 60+ native candlestick patterns
- **Main Language:** Python 3.9+ (rolling 5-version support: 3.10–3.14)
- **Coding Style:** PEP 8; type hints on all function signatures; f-strings preferred; pandas extension via `@pd.api.extensions.register_dataframe_accessor("ta")`
- **Architecture:** Modular — indicators organized by category subpackage (e.g., `pandas_ta_classic/momentum/rsi.py`); dynamic category discovery via `_meta.py`; all indicators exposed through `pandas_ta_classic` namespace
- **Testing:** pytest (primary, matches CI). Run `pytest tests/ -v` for full suite (oracle deps required; see Validation section), or the smallest relevant test module for the changed area (e.g., `pytest tests/test_indicator_momentum.py -v`). Hypothesis property-based tests also use pytest.
- **GitHub interactions:** Use the GitHub MCP server exclusively; do not use GitLens or GitKraken tools for GitHub operations. Server name: `github-mcp-server` (verify via your agent's MCP list). Use its tools for PRs, issues, reviews, and comments.
- **Commits:** Never automatically stage or commit changes; every change must be manually reviewed before being committed
- **Branches:** Name as `feat/<topic>`, `fix/<topic>`, `ci/<topic>`, `docs/<topic>`. One logical change per PR. PR title: `type(scope): short description`. Run `black --check --diff pandas_ta_classic/` and `ruff check pandas_ta_classic --select E9,F63,F7,F82` before opening. Never force-push to `main`.
- **Documentation:** Update docstrings and `docs/` when behavior, indicators, or public usage change. Docs built with Sphinx + ReadTheDocs theme + MyST Parser, deployed to GitHub Pages.
- **CHANGELOG:** Use an `[Unreleased]` section at the top of `CHANGELOG.md` for changes merged to `main` that have not yet been tagged. Every PR that lands on `main` adds its entry there. At release time, rename `[Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and tag. Keep a Changelog format.
- **Releases:** Always create annotated tags: `git tag -a X.Y.Z -m "X.Y.Z"` then `git push origin X.Y.Z`. Annotated tags carry tagger identity and timestamp needed for setuptools-scm and GitHub release attribution. Never use lightweight tags (`git tag X.Y.Z`) — they carry no metadata and produce ambiguous version strings.

## Granular Context Control

Read costs scale with context size. Only read directories relevant to the task — never the whole codebase by default.

| Task type | Read these directories / files |
|---|---|
| New indicator | `pandas_ta_classic/<category>/` (e.g., `momentum/`); existing indicator in same category for template |
| Fix indicator bug | `pandas_ta_classic/<category>/<indicator>.py` + corresponding test in `tests/` |
| Candlestick pattern (new or fix) | `pandas_ta_classic/candles/` + `pandas_ta_classic/candles/cdl_pattern.py` + `tests/test_indicator_candle.py` |
| Overlap / moving average | `pandas_ta_classic/overlap/` + `tests/test_indicator_overlap.py` |
| Strategy system | `pandas_ta_classic/core.py` + `pandas_ta_classic/custom.py` + `tests/test_strategy.py` |
| DataFrame accessor / API | `pandas_ta_classic/core.py` + `tests/test_accessor_api.py` + `tests/test_accessor_conformance.py` |
| Utilities / helpers | `pandas_ta_classic/utils/` + `tests/test_utils.py` |
| Math operators | `pandas_ta_classic/math/` + `tests/test_indicator_math.py` |
| Testing infrastructure | `tests/config.py`, `tests/context.py`, `tests/fixtures/`, `tests/assertions.py` |
| CI / workflow | `.github/workflows/` only; add `AGENTS.md` if pipeline order changes |
| Formatting / linting | `pyproject.toml` + the specific file under review |
| Docs update | `docs/` + the specific doc file; `README.md`; `AGENTS.md` only if structure changes |
| Dependency change | `pyproject.toml` |

**Rule:** If the task touches one indicator module, read that module and its test file. Pull in `pandas_ta_classic/utils/` only when shared utilities are involved. Avoid reading unrelated categories.

## Working Principles

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain or unclear or multiple interpretations exist, stop and ask — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

### Fail-Fast — No Fallbacks

**If something is wrong, raise. Never substitute a default.**

Fallbacks hide failures and produce silently wrong results. If a function can't do its job with the inputs it received, it raises — always. No hardcoded defaults, no swallowed exceptions, no plausible-looking return values on error.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- For hard problems (irreversible data, security, multi-file coordination, broad refactors), the simplest solution may not be the best solution. State the approach before implementing and list what it makes harder later.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, design smells, or architectural issues in code you're working with, mention them — don't silently work around them or fix them inline. We'll address them as a separate task.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### Challenge the Request

**You're a reasoning partner, not a code producer. Suggest materially better approaches.**

- If the requested approach risks wasted work, tech debt, broken tests, or bad architecture — stop and propose the alternative with 2-4 bullet tradeoffs.
- If a safer, simpler, or more maintainable approach achieves the same goal — flag the tradeoffs, then proceed if the user confirms.
- If the request is safe but suboptimal — flag briefly ("alternative: X would avoid Y"), then implement as asked.
- If it's purely a style preference or trivial refactor — don't interrupt.

Threshold: challenge when the alternative avoids irreversible work, security risk, data loss, broad refactors, or hours of wasted debugging. Do not challenge over minor style differences.

### Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

After each coding session, execute the code/module in local venv and troubleshoot from terminal output.

### Security

- Never commit secrets, credentials, or API keys (e.g., Alpha Vantage API keys in `utils/data/`)
- Validate external data at trust boundaries: Yahoo Finance and Alpha Vantage responses in `utils/data/`
- No arbitrary code execution from user-supplied strings

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Formatting and Linting

- **black** — formatter: `line-length=150`, `skip-string-normalization = true` (keep quotes as-is). CI runs `black --check --diff pandas_ta_classic/`. Apply locally with `black pandas_ta_classic/`. Black owns formatting.
- **ruff** — linter only (ruff format is disabled; black owns formatting). Critical checks: `--select E9,F63,F7,F82`. Advisory checks: `--extend-select C901,E501 --exit-zero`.
- Config in `pyproject.toml` under `[tool.black]` and `[tool.ruff]`
- If black and ruff format disagree on a region, lock it with `# fmt: off` / `# fmt: on`
- **Gate condition:** `black --check --diff pandas_ta_classic/` and `ruff check pandas_ta_classic --select E9,F63,F7,F82` must both return EXIT=0 before the task is considered complete. If black reports a reformat, run `black pandas_ta_classic/` then re-check.

## Imports and Paths

- Package installed in editable mode: `pip install -e .` or `uv pip install -e .`
- Module-level imports go at the top of every file; no `sys.path` manipulation
- `pandas_ta_classic/` and each category subpackage have `__init__.py`
- Cross-module imports use absolute paths: `from pandas_ta_classic.utils import get_offset`
- Within a category, relative imports use flat names: `from .sma import sma`
- Version auto-generated by setuptools-scm in `pandas_ta_classic/_version.py` (gitignored)

## CI Pipeline (6 jobs)

| Job | Description |
|---|---|
| `code-quality` | Black formatting check + Ruff linting (critical + advisory) |
| `generate-matrix` | Dynamically computes 5 supported Python versions (LATEST-4 through LATEST) |
| `testing-core` | Runs non-oracle tests on all 5 Python versions (`pytest tests/` excluding oracle suites) |
| `testing-oracle` | Runs `test_oracle_talib.py` + `test_oracle_tulipy.py` on all 5 Python versions |
| `documentation` | Builds Sphinx docs + deploys to GitHub Pages (on push only) |
| `pypi-publish` | Builds wheel, twine check, publishes to PyPI (on release published only) |

Triggers: `push` to main, `pull_request` to main, `release` published, `workflow_dispatch`.
Additional workflow: `mirror.yml` syncs repository to Codeberg on every push + nightly.

**Note:** CDL candlestick patterns have no CI tests — no CI job exercises candle pattern tests.

## Indicator Development

### File Structure

Each indicator lives in its own module under `pandas_ta_classic/<category>/<indicator>.py`:
- One public function matching the indicator name (e.g., `def rsi(...)`)
- Standard signature: `(close, length=None, ..., offset=None, **kwargs)`
- Returns: pandas Series or DataFrame

### Adding a New Indicator

1. Create `pandas_ta_classic/<category>/<indicator>.py`
2. Follow existing indicator template in same category
3. Add test in appropriate `tests/test_indicator_<category>.py` or `tests/test_ext_indicator_<category>.py`
   (overlap uses `test_ext_indicator_overlap_ext.py`; candles has no ext test file)
4. Add entry to `docs/indicators.rst`
5. Add a bullet under `### Added` in the `[Unreleased]` section of `CHANGELOG.md` describing the indicator
6. Category auto-discovery picks it up via `_meta.py` — no manual registration needed

### TA-Lib / Numba Integration

- Indicators with TA-Lib counterparts: set `_talib_module = True` and use `Imports["talib"]` or `verify_series()` helper
- Numba acceleration: use `@njit` decorator from `pandas_ta_classic.utils._njit` on hot-loop functions
- Oracle tests compare native output against TA-Lib when available

## Validation

```bash
# Create venv (first time only)
python -m venv .venv        # stdlib
# or: uv venv .venv         # faster alternative if uv installed

# Activate and install
source .venv/bin/activate
pip install -e .             # or: uv pip install -e .

# Core import check
python -c "import pandas_ta_classic; print(pandas_ta_classic.version)"

# Formatting check
black --check --diff pandas_ta_classic/

# Linting (critical errors only)
ruff check pandas_ta_classic --select E9,F63,F7,F82

# Linting (advisory — non-blocking in CI)
ruff check pandas_ta_classic --extend-select C901,E501 --exit-zero

# Apply formatting
black pandas_ta_classic/

# Install oracle dependencies (required for full test suite)
pip install -e ".[oracle]"

# Full test suite
pytest tests/ -v

# Oracle tests only
pytest tests/test_oracle_talib.py tests/test_oracle_tulipy.py -v

# Single test module (fastest feedback)
pytest tests/test_indicator_momentum.py -v

# Specific test
pytest tests/test_indicator_momentum.py::TestRSI::test_rsi -v

# Hypothesis property-based tests
pytest tests/test_property_based.py -v

# Docs build
cd docs && make html

# Build distribution
python -m build
```

## Repository Structure

```
.
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE                           # MIT
├── pyproject.toml                    # Project config, deps, tooling
├── Makefile                          # Dev task automation
├── .github/
│   ├── copilot-instructions.md          # instructs Copilot to read AGENTS.md
│   ├── dependabot.yml
│   ├── FUNDING.yml
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
│       ├── ci.yml                    # Main CI pipeline (6 jobs)
│       └── mirror.yml                # Codeberg mirror sync
├── docs/                             # Sphinx documentation
│   ├── index.rst
│   ├── conf.py
│   ├── indicators.rst                # Full indicator reference
│   ├── dataframe_api.rst
│   ├── strategies.rst
│   ├── performance.rst
│   ├── installation.rst
│   ├── quickstart.md
│   ├── tutorials.md
│   ├── testing.rst
│   └── indicator_support_matrix.rst
├── pandas_ta_classic/                # Main package source
│   ├── __init__.py
│   ├── _meta.py                      # Version + category auto-discovery
│   ├── core.py                       # AnalysisIndicators + Strategy (large file)
│   ├── custom.py                     # Custom indicator loading
│   ├── candles/                      # Candlestick patterns
│   ├── cycles/                       # Cycle indicators
│   ├── momentum/                     # Momentum indicators
│   ├── overlap/                      # Moving averages & trend-following
│   ├── performance/                  # Performance metrics
│   ├── statistics/                   # Statistical functions
│   ├── trend/                        # Trend indicators
│   ├── volatility/                   # Volatility indicators
│   ├── volume/                       # Volume indicators
│   ├── math/                         # Math operators & transforms
│   └── utils/                        # Shared utilities
│       └── data/                     # Data integrations (Alpha Vantage, Yahoo Finance)
├── tests/                            # Test suite
│   ├── config.py, context.py
│   ├── assertions.py
│   ├── fixtures/                     # expected_values.json, regression_snapshots.json
│   └── test_*.py                     # Indicator, accessor, strategy, utils tests
└── examples/                         # Jupyter notebooks & sample data
```

## Docs Summary

| File | Summary |
|---|---|
| `CHANGELOG.md` | Detailed changelog with version history, added indicators, fixes, and deprecations |
| `CONTRIBUTING.md` | Full contributor guide: dev setup, coding standards, PR process, indicator checklist |
| `docs/indicators.rst` | Complete reference of all 250+ indicators across 10 categories + 60+ CDL patterns with parameters and return types |
| `docs/indicator_support_matrix.rst` | Matrix mapping each indicator to its TA-Lib/tulipy counterpart where available |
| `docs/strategies.rst` | Strategy system documentation: multiprocessing, named groups, `df.ta.strategy()` |
| `docs/testing.rst` | Testing guide: pytest structure, oracle tests, regression snapshots, property-based tests |
| `docs/performance.rst` | Backtesting and performance metrics documentation |
