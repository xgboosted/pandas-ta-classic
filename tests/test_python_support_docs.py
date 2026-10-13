"""The supported Python range stated in the docs matches the one CI enforces.

``LATEST_PYTHON_VERSION`` in ``.github/workflows/ci.yml`` owns the window
(LATEST-4..LATEST); ``requires-python`` and the docs repeat its bounds, so
they move together or this test fails (AGENTS.md, rule 9).
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATEST = int(re.search(r'LATEST_PYTHON_VERSION: "3\.(\d+)"', (ROOT / ".github/workflows/ci.yml").read_text()).group(1))
LOW, HIGH = f"3.{LATEST - 4}", f"3.{LATEST}"
WINDOW = f"{LOW}\N{EN DASH}{HIGH}"  # the docs write the range with an en dash


def test_requires_python_is_the_window_floor():
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert requires == f">={LOW}"


@pytest.mark.parametrize("doc", ["README.md", "CONTRIBUTING.md", "AGENTS.md", "docs/index.rst", "docs/installation.rst"])
def test_doc_states_the_window(doc):
    assert WINDOW in (ROOT / doc).read_text(encoding="utf-8"), f"{doc} does not state Python {WINDOW}"
