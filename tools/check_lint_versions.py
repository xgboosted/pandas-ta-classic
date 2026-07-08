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
    pyproject_versions = {m[1]: m[2] for pin in lint_pins if (m := re.match(r"([a-zA-Z_-]+)>=(.+)", pin))}

    # The config is a repo-controlled flat file, so a regex pairing each
    # `repo:` with the `rev:` on the next line beats a PyYAML dependency.
    # Revs may be YAML-quoted, and ruff-pre-commit tags are v-prefixed
    # (v0.15.20) while black's are bare — normalize both away.
    precommit = (ROOT / ".pre-commit-config.yaml").read_text()
    precommit_versions = {
        REPO_TO_TOOL[repo]: rev.strip("'\"").lstrip("v")
        for repo, rev in re.findall(r"- repo: (\S+)\s*\n\s*rev: (\S+)", precommit)
        if repo in REPO_TO_TOOL
    }

    # Missing on either side is a failure too — `!=` alone would let a tool
    # absent from BOTH files pass as None == None.
    mismatches = [
        f"{tool}: pyproject.toml={pyproject_versions.get(tool)} vs .pre-commit-config.yaml={precommit_versions.get(tool)}"
        for tool in sorted(REPO_TO_TOOL.values())
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
