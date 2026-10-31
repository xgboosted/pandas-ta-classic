"""The counts, dependency floors and contract tests the docs repeat match their owner (AGENTS.md, rule 9).

The docs state the number of indicators, patterns, categories and TA-Lib
passthroughs about 25 times, and the pandas, numpy and TA-Lib floors in several
guides. Nothing checked them, so a new indicator or a raised floor left stale
copies behind. The registry and ``pyproject.toml`` own these facts; every copy
in the docs must match them. The list of contract tests is repeated in
AGENTS.md and docs/testing.rst, and all three copies once missed a new one.
"""

import inspect
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
# The tracked files only: a local draft or build output under docs/ is not documentation.
_TRACKED = subprocess.run(
    ["git", "ls-files", "AGENTS.md", "README.md", "CONTRIBUTING.md", "docs/*.rst", "docs/*.md"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.split()
DOCS = [ROOT / name for name in sorted(_TRACKED)]

INDICATORS = sorted({name for names in ta.Category.values() for name in names})
TALIB = [n for n in INDICATORS if n not in ta.Category["candles"] and "talib" in inspect.signature(_find_indicator_func(n)).parameters]
# cdl_doji and cdl_inside are category indicators and pattern names, so the union counts them once
UNIQUE = len(set(INDICATORS) | {f"cdl_{pattern}" for pattern in ALL_PATTERNS})

# (what the number counts, its pattern in the docs, its value)
COUNTS = [
    ("indicators", r"\b(\d+)\*{0,2} (?:category |technical )?indicators\b", len(INDICATORS)),
    ("unique indicators and patterns", r"\b(\d+) unique indicators\b", UNIQUE),
    ("CDL patterns", r"\b(\d+)\*{0,2} (?:native )?(?:cdl |candlestick )?pattern(?:s| names)\b", len(ALL_PATTERNS)),
    ("CDL patterns", r"\bCDL patterns \((\d+)\)", len(ALL_PATTERNS)),
    ("categories", r"\b(\d+) categories\b", len(ta.Category)),
    ("candle wrappers", r"\b(\d+) wrappers\b", len(ta.Category["candles"])),
    ("TA-Lib passthroughs", r"\b(\d+) non-candle indicators\b", len(TALIB)),
    ("TA-Lib passthroughs", r"(?:passthrough|Core indicators) \((\d+)\)", len(TALIB)),
]


def _stated(pattern: str) -> list[tuple[str, str]]:
    """Each (place, captured value) where a doc matches *pattern*."""
    found = []
    for path in DOCS:
        for match in re.finditer(pattern, path.read_text(encoding="utf-8"), re.IGNORECASE):
            found.append((f"{path.relative_to(ROOT)}: {match.group(0)!r}", match.group(1)))
    return found


@pytest.mark.parametrize(("what", "pattern", "value"), COUNTS, ids=[f"{what}-{i}" for i, (what, _, _) in enumerate(COUNTS)])
def test_every_stated_count_matches_the_registry(what, pattern, value):
    stated = _stated(pattern)
    assert stated, f"no doc states the number of {what}; the pattern {pattern!r} is stale"
    wrong = [where for where, number in stated if int(number) != value]
    assert not wrong, f"the number of {what} is {value}: {wrong}"


def _floors() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + [r for extra in project["optional-dependencies"].values() for r in extra]
    floors = {}
    for requirement in requirements:
        match = re.fullmatch(r"(pandas|numpy|ta-lib)>=([\d.]+)", requirement.lower())
        if match:
            floors[match.group(1)] = match.group(2)
    return floors


@pytest.mark.parametrize("package", ["pandas", "numpy", "ta-lib"])
def test_every_stated_floor_matches_pyproject(package):
    floor = _floors()[package]
    stated = _stated(rf"\b{re.escape(package)}\s*>=\s*([\d.]+\d)")
    assert stated, f"no doc states the {package} floor"
    wrong = [where for where, number in stated if number != floor]
    assert not wrong, f"pyproject.toml sets {package}>={floor}: {wrong}"


@pytest.mark.parametrize("doc", ["AGENTS.md", "docs/testing.rst"])
def test_every_contract_test_is_listed(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = sorted(path.name for path in (ROOT / "tests").glob("test_*_contract.py") if path.name not in text)
    assert not missing, f"{doc} does not name {missing}"
