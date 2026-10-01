# Candle Pattern (CDL_PATTERN)
import importlib
import os
from collections.abc import Sequence
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._core import _number, nan_on_short_input

from . import cdl_doji, cdl_inside

ALL_PATTERNS = [
    "2crows",
    "3blackcrows",
    "3inside",
    "3linestrike",
    "3outside",
    "3starsinsouth",
    "3whitesoldiers",
    "abandonedbaby",
    "advanceblock",
    "belthold",
    "breakaway",
    "closingmarubozu",
    "concealbabyswall",
    "counterattack",
    "darkcloudcover",
    "doji",
    "dojistar",
    "dragonflydoji",
    "engulfing",
    "eveningdojistar",
    "eveningstar",
    "gapsidesidewhite",
    "gravestonedoji",
    "hammer",
    "hangingman",
    "harami",
    "haramicross",
    "highwave",
    "hikkake",
    "hikkakemod",
    "homingpigeon",
    "identical3crows",
    "inneck",
    "inside",
    "invertedhammer",
    "kicking",
    "kickingbylength",
    "ladderbottom",
    "longleggeddoji",
    "longline",
    "marubozu",
    "matchinglow",
    "mathold",
    "morningdojistar",
    "morningstar",
    "onneck",
    "piercing",
    "rickshawman",
    "risefall3methods",
    "separatinglines",
    "shootingstar",
    "shortline",
    "spinningtop",
    "stalledpattern",
    "sticksandwich",
    "takuri",
    "tasukigap",
    "thrusting",
    "tristar",
    "unique3river",
    "upsidegap2crows",
    "xsidegap3methods",
]


def _discover_native_patterns() -> dict:
    """Auto-discover native cdl_*.py pattern implementations."""
    skip = {"cdl_pattern", "cdl_z", "cdl_inside", "cdl_doji"}
    native = {}
    pkg_dir = os.path.dirname(__file__)
    for fname in os.listdir(pkg_dir):
        if not fname.startswith("cdl_") or not fname.endswith(".py"):
            continue
        mod_name = fname[:-3]  # strip .py
        if mod_name in skip:
            continue
        pattern_name = mod_name[4:]  # strip cdl_
        # A broken pattern module is a bug: let it raise instead of silently
        # dropping the pattern (it used to be logged and skipped).
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        native[pattern_name] = getattr(mod, mod_name)
    return native


# Pre-built dict of native pattern name -> callable
_NATIVE_PATTERNS = _discover_native_patterns()


def _run_one_cdl_pattern(n, open_, high, low, close, pta_patterns, scalar, offset, result, **kwargs):
    """Attempt to compute and store one candle pattern by name."""
    col_name = f"CDL_{n.upper()}"
    if n in pta_patterns:
        pattern_result = pta_patterns[n](open_, high, low, close, offset=offset, scalar=scalar, **kwargs)
        if pattern_result is not None:
            result[pattern_result.name] = pattern_result
    elif n in _NATIVE_PATTERNS:
        pattern_result = _NATIVE_PATTERNS[n](open_, high, low, close, scalar=scalar, offset=offset, **kwargs)
        if pattern_result is not None:
            result[col_name] = pattern_result
    else:
        # ALL_PATTERNS is hand-maintained, _NATIVE_PATTERNS is discovered from
        # candles/cdl_*.py, and cdl_pattern() rejects a name in neither. A name
        # can only reach here if those two lists drift apart.
        raise RuntimeError(f"cdl_pattern() {n!r} is listed in ALL_PATTERNS but no module implements it")


@nan_on_short_input
def cdl_pattern(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    name: str | Sequence[str] = "all",
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Candle Pattern"""
    # Validate Arguments
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)

    if open_ is None or high is None or low is None or close is None:
        return None

    offset = get_offset(offset)
    scalar = _number(scalar, 100, "scalar")

    # Patterns with custom implementations (non-standard signatures)
    pta_patterns = {
        "doji": cdl_doji,
        "inside": cdl_inside,
    }

    if name == "all":
        name = ALL_PATTERNS
    if type(name) is str:
        name = [name]
    # a misspelled name used to be logged and skipped, returning nothing
    unknown = [n for n in name if n not in ALL_PATTERNS]
    if unknown:
        raise ValueError(f"cdl_pattern() name has unknown pattern(s): {unknown}; see ta.ALL_PATTERNS")

    result: dict = {}
    for n in name:
        _run_one_cdl_pattern(
            n,
            open_,
            high,
            low,
            close,
            pta_patterns,
            scalar,
            offset,
            result,
            **kwargs,
        )

    if not result:
        return None

    # Prepare DataFrame to return
    df = DataFrame(result)
    df.name = "CDL_PATTERN"
    df.category = "candles"
    return df


cdl_pattern.__doc__ = """Candle Pattern

A wrapper around all candle patterns.

Examples:

Get all candle patterns (This is the default behaviour)
>>> df = df.ta.cdl_pattern(name="all")
Or
>>> df.ta.cdl("all", append=True) # = df.ta.cdl_pattern("all", append=True)

Get only one pattern
>>> df = df.ta.cdl_pattern(name="doji")
Or
>>> df.ta.cdl("doji", append=True)

Get some patterns
>>> df = df.ta.cdl_pattern(name=["doji", "inside"])
Or
>>> df.ta.cdl(["doji", "inside"], append=True)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    name (str | Sequence[str]): name of the patterns
    scalar (float): How much to magnify. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: one column for each pattern.
"""

cdl = cdl_pattern
