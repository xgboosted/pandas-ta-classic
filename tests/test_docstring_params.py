"""Every parameter an indicator docstring documents is one the indicator reads.

A fresh-eyes audit found 26 documented parameters that no code read, so
passing them changed nothing and raised nothing (``**kwargs`` swallowed
them): leftovers of an older signature (``kvo`` ``long``/``length_sig``,
``stc`` ``tclen``), a wrong spelling (``open`` for ``open_``), inputs the
indicator never took (``aroon`` ``close``, ``stochrsi`` ``high``/``low``) and
EMA options copied into indicators that never reach an EMA (``sma``
``adjust``/``presma``).

A documented name is accepted when it is in the signature, is read from
``kwargs`` (``pop``/``get``/``setdefault``/``[]``/``in``), or is accepted by a
package function the indicator hands its ``kwargs`` to (``ema(close, **kwargs)``,
``attach_signals(..., kwargs=kwargs)``, ``ma()``'s ``_MA_DISPATCH[name](source, **kwargs)``),
followed recursively. A 27th, ``bias`` ``drift``, looked unread until the scan followed
that dispatch table: ``vidya`` reads it.
"""

import ast
import functools
import inspect
import re
import textwrap

import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func

_ENTRY = re.compile(r"^\s{4}(\w+)\s*\(", re.MULTILINE)
_SECTION = re.compile(r"\n(?=\w[\w ]*:\s*\n)")
_KWARG_READ = re.compile(r'kwargs(?:\.(?:pop|get|setdefault)\(|\[)"(\w+)"|"(\w+)" in kwargs')
_FILL = {"fillna", "fill_method"}  # read by apply_fill, which every indicator calls


def _documented(doc: str | None) -> set:
    names: set = set()
    for section in _SECTION.split(doc or ""):
        if section.split("\n", 1)[0].strip() in ("Args:", "Kwargs:"):
            names |= set(_ENTRY.findall(section))
    return names


def _is_kwargs(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "kwargs"


def _forwards_kwargs(call: ast.Call) -> bool:
    # ema(close, **kwargs), attach_signals(..., kwargs=kwargs), _reject_trend_reset("xsignals", kwargs)
    return any(k.arg is None or _is_kwargs(k.value) for k in call.keywords) or any(_is_kwargs(a) for a in call.args)


def _resolve(func: ast.expr, namespace: dict) -> list:
    if isinstance(func, ast.Name):
        return [namespace.get(func.id)]
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return [getattr(namespace.get(func.value.id), func.attr, None)]
    # a dispatch table: ma() calls _MA_DISPATCH[name](source, **kwargs), which may be any of its values
    if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name) and isinstance(namespace.get(func.value.id), dict):
        return list(namespace[func.value.id].values())
    return []


@functools.cache
def _accepted(fn) -> frozenset:
    fn = inspect.unwrap(fn)
    source = textwrap.dedent(inspect.getsource(fn))
    names = set(inspect.signature(fn).parameters) - {"kwargs"}
    names |= {a or b for a, b in _KWARG_READ.findall(source)}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _forwards_kwargs(node):
            for target in _resolve(node.func, fn.__globals__):
                target = inspect.unwrap(target) if callable(target) else None
                if inspect.isfunction(target) and target.__module__.startswith("pandas_ta_classic") and target is not fn:
                    names |= _accepted(target)
    return frozenset(names)


INDICATORS = sorted(i for names in ta.Category.values() for i in names)


@pytest.mark.parametrize("name", INDICATORS)
def test_documented_parameters_are_read(name):
    func = _find_indicator_func(name)
    phantom = sorted(_documented(func.__doc__) - _accepted(func) - _FILL)
    assert not phantom, f"{name}() documents parameters it never reads: {phantom}"


def test_the_scan_sees_the_parameters():
    """Guard the guard: a broken section parser would pass everything."""
    assert {"close", "length", "offset", "fillna"} <= _documented(ta.rsi.__doc__)
    assert {"xa", "signal_indicators"} <= _accepted(_find_indicator_func("rsi"))  # through attach_signals(kwargs=kwargs)
    assert "adjust" in _accepted(_find_indicator_func("tema"))  # through ema(close, **kwargs)
    assert "adjust" not in _accepted(_find_indicator_func("sma"))
    # through ma(), which looks the average up in _MA_DISPATCH: bias(mamode="vidya", drift=3) changes the result
    assert "drift" in _accepted(_find_indicator_func("bias"))
    assert "asc" in _accepted(_find_indicator_func("ma"))  # ma("wma", close, asc=False)


def test_the_scan_reports_a_phantom():
    """Guard the guard: a documented name nothing reads is reported."""

    def indicator(close, length=None, **kwargs):
        return close.rolling(length).mean()

    # laid out like the package's module-level __doc__ assignments
    indicator.__doc__ = "Args:\n    close (pd.Series): Series of 'close's\n    length (int): The period.\n    tclen (int): Read by nothing.\n"
    assert _documented(indicator.__doc__) - _accepted(indicator) == {"tclen"}


# Options the df.ta accessor reads itself, before or after the indicator runs.
_ACCESSOR = {"append", "prefix", "suffix", "col_names", "timed"}
_EXAMPLE_CALL = r"(df\.)?\bta\.{}\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
_EXAMPLE_KEYWORD = re.compile(r"(?<![\w.\"'\[])(\w+)\s*=(?!=)")


@pytest.mark.parametrize("name", INDICATORS)
def test_docstring_examples_pass_parameters_that_are_read(name):
    """A keyword in a docstring example is one the indicator reads.

    ``stc``'s examples kept ``tclen=`` after the parameter became ``tclength``,
    so the documented call ran with the default 10 without a word.
    """
    func = _find_indicator_func(name)
    unread = set()
    for accessor, call in re.findall(_EXAMPLE_CALL.format(name), func.__doc__ or ""):
        unread |= set(_EXAMPLE_KEYWORD.findall(call)) - _accepted(func) - _FILL - (_ACCESSOR if accessor else set())
    assert not unread, f"{name}() docstring examples pass parameters it never reads: {sorted(unread)}"
