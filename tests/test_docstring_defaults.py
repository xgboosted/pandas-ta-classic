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

_HELPERS = {"_pos_int", "_pos_float", "_number", "_bool_param", "_str_param"}
_PACKAGE = Path(pandas_ta_classic.__file__).parent
_MODULES = sorted(p for p in _PACKAGE.glob("*/*.py") if not p.name.startswith("_") and p.parent.name != "utils")


def _code_defaults(source: str) -> dict:
    found = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        if name in _HELPERS and len(call.args) >= 2 and isinstance(call.args[0], ast.Name):
            try:
                found.setdefault(call.args[0].id, ast.literal_eval(call.args[1]))
            except ValueError:  # computed defaults (e.g. lensig defaults to length)
                pass
        elif name == "get_drift" and call.args and isinstance(call.args[0], ast.Name):
            found.setdefault(call.args[0].id, 1)
    return found


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
    source = module.read_text()
    code, doc = _code_defaults(source), _doc_defaults(source)
    wrong = {name: (doc[name], code[name]) for name in doc.keys() & code.keys() if doc[name] != code[name]}
    assert not wrong, f"docstring Default vs code default: {wrong}"


def test_sweep_compares_a_meaningful_number_of_defaults():
    compared = sum(len(_doc_defaults(m.read_text()).keys() & _code_defaults(m.read_text()).keys()) for m in _MODULES)
    assert compared > 300
