# Agents.md

## Purpose
This file documents the use of AI coding agents and assistants for the pandas-ta-classic repository. It ensures all contributors follow consistent standards when submitting code, regardless of the agent or tool used.

## Repository Context
- **Project:** pandas-ta-classic — Technical Analysis Indicators for pandas DataFrames
- **Main Language:** Python
- **Coding Style:** Follow PEP8 and repo-specific conventions
- **Testing:** All code must pass existing tests in `tests/` and include new tests for new features or bugfixes
- **Documentation:** Update or add docstrings and docs in `docs/` as needed

## Working Principles
- Think before coding: make assumptions explicit, surface ambiguity early, and prefer clarifying tradeoffs over silent guesses
- Keep solutions simple: implement the minimum change that solves the problem and avoid speculative features or abstractions
- Make surgical changes: touch only the files and lines required for the requested work, and do not refactor unrelated code
- Work toward verifiable outcomes: define what success looks like for the change, then confirm it with the smallest relevant check

## Supported Agents
- GitHub Copilot (default)
- Claude (Anthropic)
- Other LLM-based agents (please specify in PR)

## Agent Usage Guidelines
- State which agent/code assistant was used in your PR description (e.g., Copilot, Claude, etc.)
- Keep changes focused on the issue or feature being addressed; avoid unrelated refactors
- Ensure all code matches repo style, passes relevant tests, and is well-documented
- Run the relevant validation commands below before opening a PR
- If agent uses custom instructions, skills, or modes (e.g., Caveman), mention them in the PR
- Update tests in `tests/` and documentation in `docs/` when behavior, indicators, or public usage change
- Do not submit code that violates repo license or includes proprietary/copied code
- For new agents, add a section below with name, usage, and any special notes

## Validation Commands
- Use `black .` to apply repository formatting locally
- Use `black --check --diff pandas_ta_classic/` to match the CI formatting check
- Use `ruff check pandas_ta_classic --select E9,F63,F7,F82` for critical lint errors
- Use `ruff check pandas_ta_classic --extend-select C901,E501 --exit-zero` for the broader non-blocking lint report used in CI
- Use `python -m unittest discover tests/ -v` for the full test suite, or run the smallest relevant test module for the changed area
- If documentation changes, use `cd docs && make html` to confirm the docs build successfully

## Adding New Agents
1. Add agent name and description below.
2. List any special requirements or setup steps.
3. Update this file in your PR.

---

_Example:_

### Claude (Anthropic)
- Usage: Used for code generation and review.
- Notes: Ensure code is compatible with pandas-ta-classic style and all tests pass.

### Copilot Caveman Skill
- Usage: Use for compressed, simple communication when requested.
- Notes: Default is normal Copilot unless otherwise specified.
