# AGENTS.md

## Repository Context

- **Project:** pandas-ta-classic — Community-maintained Python 3 technical analysis library with 224 indicators across 10 categories (Candles, Cycles, Momentum, Overlap, Performance, Statistics, Trend, Volatility, Volume, Math) plus 62 native candlestick patterns
- **Main Language:** Python 3.10+ (rolling 5-version support: 3.10–3.14)
- **Coding Style:** PEP 8; type hints on all function signatures; f-strings preferred; pandas extension via `@pd.api.extensions.register_dataframe_accessor("ta")`
- **Architecture:** Modular — indicators organized by category subpackage (e.g., `pandas_ta_classic/momentum/rsi.py`); dynamic category discovery via `_meta.py`; all indicators exposed through `pandas_ta_classic` namespace
- **Testing:** pytest (primary, matches CI). Run `pytest tests/ -v` for full suite (oracle deps required; see Validation section), or the smallest relevant test module for the changed area (e.g., `pytest tests/test_indicator_momentum.py -v`). Hypothesis property-based tests also use pytest.
- **GitHub interactions:** Use the GitHub MCP server exclusively; do not use GitLens or GitKraken tools for GitHub operations. Server name: `github-mcp-server` (verify via your agent's MCP list). Use its tools for PRs, issues, reviews, and comments.
- **Commits:** Never automatically stage or commit changes; every change must be manually reviewed before being committed
- **Branches:** Name as `feat/<topic>`, `fix/<topic>`, `ci/<topic>`, `docs/<topic>`. One logical change per PR. PR title: `type(scope): short description`. Run `black --check --diff pandas_ta_classic/`, `ruff check .` and `make typecheck` before opening. Never force-push to `main`.
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
| Testing infrastructure | `tests/config.py`, `tests/fixtures/`, `tests/assertions.py` |
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

- Never commit secrets, credentials, or API keys (e.g., a data-provider API key in an example)
- Validate external data at trust boundaries: OHLCV DataFrames from user code or examples
- No arbitrary code execution from user-supplied strings

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Formatting and Linting

- **black** — formatter: `line-length=150`, `skip-string-normalization = true` (keep quotes as-is). CI runs `black --check --diff pandas_ta_classic/`. Apply locally with `black pandas_ta_classic/`. Black owns formatting.
- **ruff** — linter only (ruff format is disabled; black owns formatting). Gate: `ruff check .` over the whole repo, using ruff's default rule set plus `ICN` from `[tool.ruff.lint]` (`numpy` is always `np.`, never `from numpy import`). The `>=` pin floats, so new default rules are adopted, not suppressed: fix the code. Library code has no per-file ignores; an unavoidable exception gets an inline `# noqa: <rule>` with a reason. Advisory: `ruff check pandas_ta_classic --extend-select C901,E501 --exit-zero`.
- **`--select` replaces the configured rule set, it does not add to it.** The gate is a bare `ruff check .`, which reads `[tool.ruff.lint]`. A command such as `ruff check --select E9,F63,F7,F82` silently skips every other rule (`F403`, `E402`, `ICN`, `UP`, `I`, ...); those four are already part of the default set.
- **pre-commit runs the same gates as CI.** `pre-commit install` installs commit and push hooks: black, ruff and the `core.pyi` regeneration run on commit, `make typecheck` runs on push. The `ruff-check` hook reads `[tool.ruff.lint]` with no `--select` and no package-only `files:` filter. Black stays package-only in pre-commit, matching CI's `black --check pandas_ta_classic/`.
- **Generated `core.pyi`** goes through **gen → `ruff check --fix` → black**, so the stub matches lint-fixed source. That pipeline lives in three places that must stay identical: the CI `code-quality` job (followed by `git diff --exit-code`), the `gen-core-stub` hook in `.pre-commit-config.yaml`, and any manual regeneration. Changing `tools/gen_core_stub.py` or the pipeline means updating all three.
- Config in `pyproject.toml` under `[tool.black]`, `[tool.ruff]`, `[tool.ruff.lint]` and `[tool.ruff.lint.isort]` (`combine-as-imports = true`)
- **mypy** — `make typecheck` (targets Python 3.12: numpy >= 2.5 stubs cannot be parsed below it); blocking in CI. It runs two passes: the package, then `pandas_ta_classic/core.py` on its own, because `core.pyi` (the IDE stub for `df.ta`) shadows `core.py` during package discovery.
- If black and ruff format disagree on a region, lock it with `# fmt: off` / `# fmt: on`
- **Gate condition:** `black --check --diff pandas_ta_classic/`, `ruff check .` and `make typecheck` must all return EXIT=0 before the task is considered complete. If black reports a reformat, run `black pandas_ta_classic/` then re-check.
- **Dual config pattern:** black/ruff versions appear in two places — `pyproject.toml` under `[project.optional-dependencies].lint` (CI installs via `pip install -e ".[lint]"`) AND `.pre-commit-config.yaml` under each hook's `rev`. When bumping a version, update BOTH. Enforced: `tools/check_lint_versions.py` (run by the CI `code-quality` job and `make lint`) fails on any mismatch. Because the pins are `>=` floors, it also compares the *installed* tools with the pre-commit `rev` by release series (ruff `0.MINOR`, black's year): when a new series is out, CI fails with instructions to bump both files, so pre-commit and CI never enforce different rules. Patch releases within a series pass.

## Imports and Paths

- Package installed in editable mode: `pip install -e .` or `uv pip install -e .`
- Module-level imports go at the top of every file; no `sys.path` manipulation
- `pandas_ta_classic/` and each category subpackage have `__init__.py`
- Cross-module imports use absolute paths: `from pandas_ta_classic.utils import get_offset`
- Within a category, relative imports use flat names: `from .sma import sma`
- Version auto-generated by setuptools-scm in `pandas_ta_classic/_version.py` (gitignored)

### Import Conventions

Repo-wide. Each rule is marked *enforced* (ruff fails the build) or *convention* (review and the checklist below).

- **numpy is `import numpy as np`, used as `np.sqrt`, `np.nan`, `np.lib.stride_tricks.sliding_window_view`.** *Enforced:* ruff `ICN001` rejects any other alias, and `ICN003` with `banned-from = ["numpy", "numpy.lib.stride_tricks"]` rejects `from numpy import x` and `from numpy.lib.stride_tricks import x` in every form, including the old `npX` aliases (`npNaN`, `nplog`, `npsqrt`).
- **pandas is `from pandas import Series, DataFrame`.** *Convention.* pandas is deliberately not in `banned-from`: indicator signatures read `def rsi(close: Series, ...)` at 300+ sites. `import pandas as pd` (used in `core.py` and tests) only has its alias checked, by `ICN001`.
- **No star imports.** Re-exports are explicit with an `__all__` list (`pandas_ta_classic/__init__.py`, `utils/__init__.py`); category subpackages get `__all__` at runtime from `_lazy_subpackage.install_lazy_subpackage()`. *Enforced:* `F403`.
- **Function-body imports are for optional dependencies only** (`talib`, `tulipy`, `tqdm`, lazy-loading machinery). numpy and pandas are hard dependencies and are imported at module scope. *Convention:* ruff's `PLC0415` would also flag the intentional optional-dependency imports, so this is checked by grep instead.
- **No `sys.path` bootstrapping** in `pandas_ta_classic/`, `tests/`, `tools/` or `docs/`; the editable install and pytest's rootdir handling make it redundant. `custom.py` is the exception: it adds the user's own indicator directory at runtime. *Convention.*
- **`logger = logging.getLogger(__name__)` goes after the import block.** *Enforced:* `E402`.
- **Prefer stdlib over hand-rolled math**, e.g. `combination()` wraps `math.comb`. Do not reintroduce hand-rolled nCr, erf, factorial or gcd loops. *Convention.*
- **Dead code goes.** A helper with no callers is deleted, together with the constants and imports only it used. Search the whole repo, including the defining file, before deciding. *Convention.*
- **Validate numeric parameters with `_pos_int` / `_pos_float` / `_number`** (`utils/_core.py`), e.g. `length = _pos_int(length, 10, "length")`, `_pos_float(na, 0.2, "na", lt=1)` or `scalar = _number(scalar, 100, "scalar")` (any finite number); `drift` and `offset` go through `get_drift` / `get_offset`; flags use `_bool_param(talib, False, "talib")` and string options `_str_param(method, "classic", "method", choices={...})`. `None` selects the default; any other invalid value (0, negative, NaN, bool, a fractional int) raises `ValueError` naming the indicator and parameter. Never write `x = int(x) if x and x > 0 else default`, which silently replaces bad input. If an indicator passes its value to another indicator with a stricter bound, give it the same bound so the error names the function the user called. *Convention:* greps 5–8 below.
- **No import cruft.** No `pkg_resources` fallbacks (setuptools-scm owns the version) and no commented-out import lines; ruff reads code, not comments. *Convention.*

#### Convention checklist (run before finishing)

The enforced rules are covered by the **Gate condition** above. The conventions have no CI backstop, so run these greps before considering a task complete. **Each must print nothing**; a hit is a violation to fix or to justify in review.

```bash
# 1. numpy/pandas imported inside a function or block (must be module scope)
grep -rnE "^ +(import (numpy|pandas)\b|from (numpy|pandas) import)" pandas_ta_classic/ tests/ tools/ examples/ docs/ --include=*.py

# 2. sys.path writes outside custom.py
grep -rnE "sys\.path\.(insert|append)" pandas_ta_classic/ tests/ tools/ docs/ | grep -v "custom.py"

# 3. Import cruft: pkg_resources fallbacks and commented-out numpy/pandas imports
grep -rnE "pkg_resources|^\s*#\s*(from|import) (numpy|pandas)" pandas_ta_classic/ tests/ tools/ examples/ docs/ --include=*.py

# 4. Hand-rolled math that stdlib covers (nCr loop, Abramowitz-Stegun erf constant)
grep -rnE "reduce\(mul|numerator // denominator|0\.3275911" pandas_ta_classic/

# 5. Numeric parameters validated by a silent-default guard instead of _pos_int/_pos_float/_number
grep -rnE "^\s+(\w+) = ((int|float)\()?\1\)? if (\1 and \1 [<>]|\1 else|\1 is not None|isinstance\(\1)" pandas_ta_classic/

# 6. Silent-default guards grep 5 misses: is_percent(), membership and range tests, bool()/abs()
#    coercion, int(kwargs[...]) (any "x = <x transformed> if <test on x> else <default>")
grep -rnE "^\s+(\w+) = .*\b\1\b.* if .*\b\1\b.* else |int\(kwargs\[|= int\(abs\(" pandas_ta_classic/ --include=*.py

# 7. Numeric options read from **kwargs without validation (wrap them: _pos_int(kwargs.pop("x", None), 5, "x")).
#    Signal thresholds xa/xb are checked once, in utils/_signals.py signals().
grep -rnE 'kwargs\.(pop|get)\("\w+", -?[0-9.]+\)' pandas_ta_classic/ --include=*.py | grep -vE '\bx[ab]=kwargs'

# 8. True/False options read from **kwargs by truthiness (wrap them: _bool_param(kwargs.pop("x", None), False, "x")).
#    stc's ma1/ma2/osc default to False but take Series; msw's tulipy flag is handled in its own module.
grep -rnE 'kwargs\.(pop|get)\("\w+", (True|False)\)' pandas_ta_classic/ --include=*.py | grep -vE '"(ma1|ma2|osc|tulipy)"'
```

Not greppable, so check in review: **dead code** your change orphaned, and the **stdlib over hand-rolled** preference.

## CI Pipeline (6 jobs)

| Job | Description |
|---|---|
| `code-quality` | Black formatting check, `ruff check .` (blocking) + advisory ruff, mypy, core.pyi sync, lint-version parity |
| `generate-matrix` | Dynamically computes 5 supported Python versions (LATEST-4 through LATEST) |
| `testing-core` | Runs non-oracle tests on all 5 Python versions (`pytest tests/` excluding oracle suites) |
| `testing-oracle` | Runs `test_oracle_talib.py` + `test_oracle_tulipy.py` on all 5 Python versions |
| `documentation` | Builds Sphinx docs + deploys to GitHub Pages (on push only) |
| `pypi-publish` | Builds wheel, twine check, publishes to PyPI (on release published only) |

Triggers: `push` to main, `pull_request` to main, `release` published, `workflow_dispatch`.
Additional workflow: `mirror.yml` syncs repository to Codeberg on every push + nightly.

**Note:** CDL candlestick pattern tests (`test_indicator_candle.py`, `test_ext_indicator_candle.py`) run in `testing-core`; they need no TA-Lib.

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

## Correctness Rules

General rules taken from #142 (lint modernization, import conventions, silent-failure fixes, data-fetching removal), its review, and the repo-wide sweeps that followed. Each rule names what enforces it; a rule marked *review* has no automated check.

### Arguments

1. **No silent fallbacks for arguments.** `None` selects the default; any other value the function cannot use raises `ValueError` naming the function, the parameter and the value. Never write `x = int(x) if x and x > 0 else default`, `x if x else default`, `bool(x) if isinstance(x, bool) else default` or an `if x not in choices: x = default` block. Use `_pos_int`, `_pos_float`, `_number`, `_bool_param`, `_str_param`, `get_drift` and `get_offset` from `utils/_core.py`. *Enforced:* `tests/test_parameter_validation.py`, checklist greps 5–8.
2. **Give a parameter the bound its callees need.** When an indicator passes a value to another indicator with a stricter bound (`stdev` → `variance` needs `length > 1`), declare the same bound so the error names the function the user called.
3. **Helpers reject unknown keywords.** Utility functions take explicit keyword-only options (`def fibonacci(n=2, *, zero=False, weighted=False)`), never `**kwargs` read with `kwargs.pop`, so a typo raises `TypeError` instead of returning the default result. Indicators keep `**kwargs` for `fillna`/`fill_method` and for strategy-wide arguments they deliberately absorb (for example `length` in `adosc`); document such absorption in a comment. *Enforced:* `test_helpers_reject_misspelled_keywords`.
4. **Every accepted parameter is read.** A parameter the body never uses (the old `ticker(ds=...)`, `slope(vertical=...)`) is removed, or implemented. *Review.*
5. **Choices are validated where they are chosen.** `ma()` raises on an unknown name instead of returning an EMA; `mamode` and similar options inherit that. *Enforced:* `test_behaviour_fixes_from_the_same_sweep`.

### TA-Lib paths

6. **`talib=True` honours every parameter or does not run.** The TA-Lib branch requires every parameter TA-Lib cannot express to be at its default (`if Imports["talib"] and mode_talib and scalar == 100 and drift == 1:`); any other value computes natively. Both paths accept the same value ranges (`mama` limits `< 1`). *Enforced:* `tests/test_talib_parameters.py`.
7. **Tests that need TA-Lib skip without it.** `talib=True` silently falls back to the native formula when TA-Lib is absent, so an oracle comparison that calls it must be guarded with `skipUnless(ta.Imports["talib"])`. *Review.*

### Documentation

8. **Docstring defaults match the code.** Every `Default:` in an indicator docstring equals the value the validation call resolves `None` to. *Enforced:* `tests/test_docstring_defaults.py`.
9. **Duplicated facts have one owner or a parity check.** Tool versions (`pyproject.toml` floor, pre-commit `rev`, installed release series), the `core.pyi` pipeline (CI, pre-commit, manual), and indicator/pattern counts (224 / 62) must agree. *Enforced:* `tools/check_lint_versions.py`, the CI stub sync; counts by *review*.
10. **Docs, examples and notebooks move with the API.** A removal or rename also updates `docs/`, `README.md`, `examples/` (scripts and notebooks), tests, `pyproject.toml` extras, `Imports` keys and mypy overrides, and leaves a removal note where the old API was documented. *Review;* the checklist greps catch leftover imports.

### Deprecation, removal and change records

11. **Deprecate in a released version before removing.** A warning that has not shipped in a tagged release is not a deprecation. Documentation names the version that shipped it (`.. deprecated:: 0.8.32`), never a guessed next version. Announce removals with a plain `.. note::`: `.. versionremoved::` needs Sphinx ≥ 7.3 and the docs floor is lower.
12. **Follow the removal plan the warning promised.** If a warning says "the default changes in the next breaking release", that release makes the change (`ichimoku` → DataFrame in 0.9.0) and the old form warns until its announced removal.
13. **Mark every breaking entry.** A `CHANGELOG.md` entry that changes results or starts raising for existing calls starts with **BREAKING**, even under Fixed.
14. **Releases use annotated tags** (`git tag -a X.Y.Z -m "X.Y.Z"`). `0.6.52` and `0.8.32` are lightweight tags; do not repeat that.

### Refactors and verification

15. **Prove a behaviour-preserving change.** Hash the raw output of every registered indicator and all candle patterns before and after, with numba and with `NUMBA_DISABLE_JIT=1`; the hashes must be identical. An automated rewrite that can touch numerics (for example ruff's `min`/`max` clamp fix inside `@njit` code) also needs its edge cases checked (NaN, ±0.0, inf).
16. **Re-run mechanical rewrites, don't rebase them.** An autofix over hundreds of files is a command, not a diff: land the decisions first, then run the tool on current `main` in its own PR.
17. **One logical change per PR.** Keep mechanical churn out of PRs that carry decisions, so the decisions stay reviewable.
18. **Type-check what ships.** A stub must not hide its implementation from mypy (`make typecheck` runs a second pass on `core.py` because `core.pyi` shadows it). Annotations use PEP 604 everywhere, including generated stubs and docstrings, and a parameter's type and the return type stay consistent (`float`, not `int | float`). *Enforced:* `make typecheck`, ruff `UP`/`PYI041`.
19. **Never mutate shared or caller-owned state.** Work on copies of shared registries (`Category`) and of the caller's objects (`Strategy.ta`). *Enforced:* `TestStrategyDoesNotMutateInputs`.

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

# Linting (blocking gate: ruff default rules + ICN)
ruff check .

# Type checking (blocking gate)
make typecheck

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
├── tests/                            # Test suite
│   ├── config.py
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
| `docs/indicators.rst` | Complete reference of all 224 indicators across 10 categories + 62 CDL patterns with parameters and return types |
| `docs/indicator_support_matrix.rst` | Matrix mapping each indicator to its TA-Lib/tulipy counterpart where available |
| `docs/strategies.rst` | Strategy system documentation: multiprocessing, named groups, `df.ta.strategy()` |
| `docs/testing.rst` | Testing guide: pytest structure, oracle tests, regression snapshots, property-based tests |
| `docs/performance.rst` | Backtesting and performance metrics documentation |
