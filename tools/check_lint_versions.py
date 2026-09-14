#!/usr/bin/env python
"""Verify black/ruff versions agree across pyproject.toml, pre-commit and the environment.

pyproject.toml's [project.optional-dependencies].lint pins and
.pre-commit-config.yaml's hook `rev:` fields are two independent copies of
the same version (see AGENTS.md's "Dual config pattern"). Nothing else
checks these agree, so a bump to one without the other would otherwise go
unnoticed until a contributor's local pre-commit run used a different
version than CI.

The lint pins are `>=` floors, so the environment (CI included) installs the
newest release. The installed version is therefore also compared with the
pre-commit rev, by release series: ruff adds default rules in 0.MINOR
releases and black changes its stable style once a year (the major
component), so a newer series means CI enforces rules pre-commit does not.
Patch releases within a series pass.
"""

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

ROOT = Path(__file__).parent.parent

# Repos whose pre-commit `rev:` mirrors a pin in the lint extra. Any other
# repo in .pre-commit-config.yaml (local hooks, future additions) is ignored
# here — add it to this map if its version also gets pinned in pyproject.toml.
REPO_TO_TOOL = {
    "https://github.com/psf/black": "black",
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
}

# Leading version components that identify a release series whose rules or
# style can differ: ruff 0.16.x -> "0.16", black 26.5.1 -> "26".
SERIES_PARTS = {"black": 1, "ruff": 2}


def _series(tool: str, ver: str) -> str:
    return ".".join(ver.split(".")[: SERIES_PARTS[tool]])


def main() -> int:
    with open(ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
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

    drifted = []
    for tool in sorted(REPO_TO_TOOL.values()):
        try:
            installed = version(tool)
        except PackageNotFoundError:
            print(f"{tool}: not installed here, skipping the installed-version check")
            continue
        rev = precommit_versions[tool]
        if _series(tool, installed) != _series(tool, rev):
            drifted.append(f"{tool}: installed {installed} vs .pre-commit-config.yaml rev {rev}")
    if drifted:
        print("Installed lint tools are a newer release series than pre-commit, so CI and pre-commit enforce different rules.")
        print("Bump the `rev:` in .pre-commit-config.yaml and the `>=` floor in pyproject.toml to the installed version:")
        for d in drifted:
            print(f"  {d}")
        return 1
    print("black/ruff versions match across pyproject.toml, .pre-commit-config.yaml and the installed tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
