"""CHANGELOG.md and the fragments in changelog.d/ that towncrier builds it from.

Each PR adds ``changelog.d/<PR number>.<type>.md`` (changelog.d/README.md has the
rules); ``towncrier build`` writes them into a new CHANGELOG.md section at release.
These tests check what towncrier does not: that a fragment is a short entry in the
form the template renders, that a breaking one is marked as AGENTS.md rule 13 asks,
and that the built file keeps the marker towncrier inserts after.

A title repeated within one version is the trace of an entry that reached the file
twice, which a hand edit at release (merging a built section into an existing one)
or a ``union`` merge of an older branch can still produce.
"""

import re
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
TOWNCRIER = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["towncrier"]
FRAGMENTS = ROOT / TOWNCRIER["directory"]
NOT_FRAGMENTS = {"README.md", Path(TOWNCRIER["template"]).name}
TYPES = {t["directory"] for t in TOWNCRIER["type"]}
MAX_LENGTH = 1000


def _fragments():
    return sorted(p for p in FRAGMENTS.iterdir() if p.name not in NOT_FRAGMENTS)


def test_no_entry_title_repeats_within_a_version():
    for section in re.split(r"^(?=## \[)", CHANGELOG.read_text(encoding="utf-8"), flags=re.MULTILINE):
        titles = Counter(re.findall(r"^\* (\*\*.+?\*\*)", section, flags=re.MULTILINE))
        repeated = sorted(title for title, count in titles.items() if count > 1)
        assert not repeated, f"{section.splitlines()[0]} repeats {len(repeated)} entry title(s): {repeated}"


def test_towncrier_marker_precedes_every_version():
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = TOWNCRIER["start_string"]
    assert text.count(marker) == 1, f"CHANGELOG.md must hold {marker!r} exactly once"
    assert text.index(marker) < text.index("\n## ["), "towncrier inserts after the marker, so it must precede the newest version"


def test_fragment_names():
    numbered = re.compile(rf"\d+\.({'|'.join(TYPES)})(\.\d+)?\.md")
    upgrading = re.compile(r"\+[\w-]+\.upgrading\.md")
    bad = [p.name for p in _fragments() if not (numbered.fullmatch(p.name) or upgrading.fullmatch(p.name))]
    assert not bad, f"name fragments <PR number>.<type>.md with a type from {sorted(TYPES)} (changelog.d/README.md): {bad}"


def test_fragment_content():
    problems = []
    for path in _fragments():
        if path.name.endswith(".upgrading.md"):
            continue  # the maintainer's release summary: a list, longer than one entry
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        if not text:
            problems.append(f"{path.name}: empty")
        elif len(text) > MAX_LENGTH:
            problems.append(f"{path.name}: {len(text)} characters, at most {MAX_LENGTH}")
        if text.startswith(("* ", "- ")):
            problems.append(f"{path.name}: starts with a bullet, which towncrier adds")
        if any(line[:1].isspace() for line in lines[1:]):
            problems.append(f"{path.name}: an indented line, which towncrier indents again")
        if "BREAKING" in text and not text.startswith("**BREAKING"):
            problems.append(f"{path.name}: a breaking entry starts with **BREAKING (AGENTS.md rule 13)")
    assert not problems, "\n".join(problems)
