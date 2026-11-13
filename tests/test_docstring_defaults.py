"""Every ``Default:`` in an indicator docstring matches the default the code applies.

The #142 review found docs contradicting code; a sweep then found 20 indicator
docstrings that stated the wrong default (``mom``/``roc``/``dpo`` documented
``length`` 1 but use 10/10/20, ``adosc`` fast/slow 12/26 but 3/10). The code
default is read from the validation call that resolves ``None``
(``length = _pos_int(length, 10, "length")``) and compared with the literal at
the start of the docstring's ``Default:``.
"""

import ast
import re
from pathlib import Path

import pytest

import pandas_ta_classic
from pandas_ta_classic.utils._keywords import _EMA as _EMA_KEYWORDS, _KWARGS_OPTIONS
from tests.contracts import code_defaults as _code_defaults

_PACKAGE = Path(pandas_ta_classic.__file__).parent
_MODULES = sorted(p for p in _PACKAGE.glob("*/*.py") if not p.name.startswith("_") and p.parent.name != "utils")


def _doc_defaults(source: str) -> dict:
    doc = re.search(r'__doc__ = """(.*?)"""', source, re.DOTALL)
    found = {}
    if doc:
        for entry in re.finditer(r"^ {4}(\w+) \([^)]*\):(.*?)(?=^ {4}\w+ \(|^\S|\Z)", doc.group(1), re.DOTALL | re.MULTILINE):
            default = re.search(r"Default:\s*('[^']*'|\"[^\"]*\"|-?[\d.]+|True|False|None)", entry.group(2))
            if default:
                found.setdefault(entry.group(1), ast.literal_eval(default.group(1)))
    return found


@pytest.mark.parametrize("module", _MODULES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_docstring_defaults_match_code(module):
    source = module.read_text(encoding="utf-8")
    code, doc = _code_defaults(source), _doc_defaults(source)
    wrong = {name: (doc[name], code[name]) for name in doc.keys() & code.keys() if doc[name] != code[name]}
    assert not wrong, f"docstring Default vs code default: {wrong}"


def test_sweep_compares_a_meaningful_number_of_defaults():
    sources = (m.read_text(encoding="utf-8") for m in _MODULES)
    compared = sum(len(_doc_defaults(source).keys() & _code_defaults(source).keys()) for source in sources)
    assert compared > 300


_EMA = _PACKAGE / "overlap" / "ema.py"
# The keyword table lists every indicator that hands ema's options on through
# **kwargs (tests/test_keyword_table.py keeps it in step with the code).
_EMA_FORWARDERS = [m for m in _MODULES if m != _EMA and _KWARGS_OPTIONS.get(m.stem, frozenset()) >= _EMA_KEYWORDS]


def test_ema_forwarders_found():
    assert {m.stem for m in _EMA_FORWARDERS} == {"dema", "massi", "t3", "tema", "trix", "tsi"}


@pytest.mark.parametrize("module", _EMA_FORWARDERS, ids=lambda p: p.stem)
def test_ema_forwarders_document_the_ema_options(module):
    """An indicator that hands **kwargs to its ema() calls takes ema's own options,
    so it documents them with ema's defaults. t3 and tema documented adjust with
    the wrong default (True) and left sma out; massi, trix and tsi documented
    neither."""
    options = {k: v for k, v in _code_defaults(_EMA.read_text(encoding="utf-8")).items() if k not in ("length", "talib")}
    doc = _doc_defaults(module.read_text(encoding="utf-8"))
    assert {k: doc.get(k) for k in options} == options
