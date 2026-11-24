"""``ta.defaults()``: an indicator's declared defaults and bounds, as data."""

from collections.abc import Callable
from typing import Any

from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.utils._core import Bound, Declared, OneOf, _require_input, declared_params


def _describe(declared: Declared) -> dict[str, Any]:
    marker = declared.marker
    if isinstance(marker, Bound):
        entry: dict[str, Any] = {
            "kind": "int" if declared.integer else "float",
            "default": declared.default,
            "gt": marker.gt,
            "ge": marker.ge,
            "lt": marker.lt,
        }
        if marker.default_from is not None:
            entry["default_from"] = marker.default_from
    elif isinstance(marker, OneOf):
        entry = {"kind": "str", "default": declared.default, "choices": marker.choices}
    else:
        entry = {"kind": "bool", "default": declared.default}
    entry["in_kwargs"] = declared.in_kwargs
    return entry


def defaults(indicator: str | Callable) -> dict[str, dict[str, Any]]:
    """{parameter: description} of the parameters an indicator declares, without running it.

    ``ta.defaults("hma")["length"]`` gives ``{"kind": "int", "default": 10, "gt": 1,
    "ge": None, "lt": None, "in_kwargs": False}``. A ``str`` option lists its
    ``choices``; a default taken from another parameter (adx: ``lensig``) names it
    under ``default_from``; ``in_kwargs`` marks an option read from ``**kwargs``.
    A bound that depends on another parameter (squeeze: ``kc_length > 1`` with
    ``lazybear``) is checked in the body and not listed.
    """
    _require_input(indicator, "defaults", "indicator")
    fn = _find_indicator_func(indicator) if isinstance(indicator, str) else indicator
    if fn is None:
        raise ValueError(f"defaults() indicator must be the name of an indicator, got {indicator!r}")
    found = declared_params(fn)
    if not found:
        raise ValueError(f"defaults() {getattr(fn, '__name__', fn)!r} declares no parameters")
    return {name: _describe(spec) for name, spec in found.items()}
