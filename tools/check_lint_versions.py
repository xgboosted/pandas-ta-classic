#!/usr/bin/env python
"""Verify black/ruff versions match between pyproject.toml and .pre-commit-config.yaml.

pyproject.toml's [project.optional-dependencies].lint pins and
.pre-commit-config.yaml's hook `rev:` fields are two independent copies of
the same version (see AGENTS.md's "Dual config pattern"). Nothing else
checks these agree, so a bump to one without the other would otherwise go
unnoticed until a contributor's local pre-commit run used a different
version than CI.
"""

import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

# Repos whose pre-commit `rev:` mirrors a pin in the lint extra. Any other
# repo in .pre-commit-config.yaml (local hooks, future additions) is ignored
# here — add it to this map if its version also gets pinned in pyproject.toml.
REPO_TO_TOOL = {
    "https://github.com/psf/black": "black",
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
}


def main() -> int:
    pyproject = tomllib.load(open(ROOT / "pyproject.toml", "rb"))
    lint_pins = pyproject["project"]["optional-dependencies"]["lint"]
    # A pin that doesn't match (or a missing tool) leaves its entry unset,
    # which the mismatch check below reports as `tool: ...=None` and fails.
    pyproject_versions = {}
    for pin in lint_pins:
        match = re.match(r"([a-zA-Z_-]+)>=(.+)", pin)
        if match:
            pyproject_versions[match.group(1)] = match.group(2)

    precommit = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    precommit_versions = {}
    for entry in precommit.get("repos", []):
        tool = REPO_TO_TOOL.get(entry.get("repo", ""))
        if tool is not None:
            # ruff-pre-commit tags are v-prefixed (v0.15.20); black's are bare.
            precommit_versions[tool] = str(entry.get("rev", "")).lstrip("v")

    # Missing on either side is a failure too — `!=` alone would let a tool
    # absent from BOTH files pass as None == None.
    mismatches = [
        f"{tool}: pyproject.toml={pyproject_versions.get(tool)} vs .pre-commit-config.yaml={precommit_versions.get(tool)}"
        for tool in ("black", "ruff")
        if pyproject_versions.get(tool) != precommit_versions.get(tool) or pyproject_versions.get(tool) is None
    ]
    if mismatches:
        print("Version mismatch between pyproject.toml and .pre-commit-config.yaml:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print("black/ruff versions match across pyproject.toml and .pre-commit-config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
