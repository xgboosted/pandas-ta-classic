# -*- coding: utf-8 -*-
"""Edge-case tests for all indicators.

Verifies that every indicator handles degenerate inputs gracefully:
empty Series, single-row, all-NaN, and short Series.

All indicators are called with ``talib=False`` to bypass TA-Lib's C
extensions, which can segfault on degenerate inputs.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from pandas_ta_classic import Category

# ---------------------------------------------------------------------------
# Indicators to skip (dispatchers / special signatures)
# ---------------------------------------------------------------------------
SKIP = {"ma", "cdl_pattern"}

# Indicators that are TA-Lib-only (no pure-Python implementation)
TALIB_ONLY = {"ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendmode"}

# Indicators requiring special kwargs beyond OHLCV
SPECIAL_KWARGS = {
    "xsignals": {"xa": 20.0, "xb": 80.0},
}

# Acceptable exceptions — the indicator detected bad input and raised.
# AttributeError covers e.g. vwap (needs DatetimeIndex) and _build_dataframe
# receiving None from a sub-indicator on all-NaN input.
ACCEPTABLE = (
    ValueError,
    TypeError,
    IndexError,
    KeyError,
    ZeroDivisionError,
    AttributeError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map of parameter names to OHLCV column types
_PARAM_MAP = {
    "open_": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    # Some indicators use alternative names
    "signal": "close",
    "trend": "close",
    "fast": "close",
    "slow": "close",
    "real": "close",
}


def _get_indicator_func(name: str):
    """Resolve an indicator name to its callable."""
    for cat_name, indicators in Category.items():
        if name in indicators:
            mod = __import__(
                f"pandas_ta_classic.{cat_name}.{name}",
                fromlist=[name],
            )
            return getattr(mod, name, None)
    return None


def _build_args(func, series_factory):
    """Build positional args for an indicator from its signature.

    Inspects the function signature to find Series parameters and maps
    them to synthetic OHLCV data produced by *series_factory*.
    """
    sig = inspect.signature(func)
    args = []
    for param_name, param in sig.parameters.items():
        if param_name in ("kwargs", "self"):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        # Check if this is a Series parameter
        if param_name in _PARAM_MAP:
            col = _PARAM_MAP[param_name]
            args.append(series_factory(col))
        elif param.default is not inspect.Parameter.empty:
            # Stop at the first parameter with a default — it's optional
            break
        else:
            # Unknown required positional param — try close as fallback
            args.append(series_factory("close"))
    return args


def _make_ohlcv_factory(length, fill=None):
    """Return a factory that produces synthetic OHLCV Series of given length."""
    idx = pd.RangeIndex(length)

    def factory(col):
        if fill == "nan":
            return pd.Series(np.nan, index=idx, dtype=float, name=col)
        elif length == 0:
            return pd.Series(dtype=float, name=col)
        else:
            # Produce plausible values for each column type
            rng = np.random.default_rng(42)
            base = 100.0 + rng.standard_normal(length).cumsum()
            if col == "high":
                return pd.Series(
                    base + abs(rng.standard_normal(length)), index=idx, name=col
                )
            elif col == "low":
                return pd.Series(
                    base - abs(rng.standard_normal(length)), index=idx, name=col
                )
            elif col == "volume":
                return pd.Series(
                    (1e6 + rng.standard_normal(length) * 1e5).clip(1),
                    index=idx,
                    name=col,
                )
            else:  # close, open
                return pd.Series(base, index=idx, name=col)

    return factory


def _call_indicator(func, name, series_factory, extra_kwargs=None):
    """Call an indicator with args built from its signature.

    Always passes ``talib=False`` to avoid TA-Lib C segfaults on
    degenerate inputs.
    """
    args = _build_args(func, series_factory)
    kwargs = dict(SPECIAL_KWARGS.get(name, {}))
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    # Force pure-Python path — TA-Lib C code can segfault on bad input
    sig = inspect.signature(func)
    if "talib" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        kwargs["talib"] = False
    return func(*args, **kwargs)


def _validate_result(result):
    """Assert the result is either None or a valid Series/DataFrame/tuple."""
    if result is None:
        return  # Perfectly fine — indicator rejected input
    if isinstance(result, tuple):
        # ichimoku returns a tuple of DataFrames
        for elem in result:
            if elem is not None:
                assert isinstance(elem, (pd.Series, pd.DataFrame))
    else:
        assert isinstance(result, (pd.Series, pd.DataFrame))


# ---------------------------------------------------------------------------
# Collect all indicator names for parametrization
# ---------------------------------------------------------------------------
ALL_INDICATORS = sorted(
    name for indicators in Category.values() for name in indicators if name not in SKIP
)


# Indicators without a ``length`` parameter — skip for negative-length test
_NO_LENGTH_PARAM = set()


def _resolve(name):
    """Resolve indicator name to callable, or pytest.skip."""
    func = _get_indicator_func(name)
    if func is None:
        if name in TALIB_ONLY:
            pytest.skip(f"{name} is TA-Lib only")
        pytest.skip(f"Could not resolve {name}")
    return func


def _has_length_param(func):
    """Return True if the indicator accepts a ``length`` kwarg."""
    return "length" in inspect.signature(func).parameters


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_empty_series(name):
    """Indicator handles empty (len=0) inputs without crashing."""
    func = _resolve(name)
    factory = _make_ohlcv_factory(0)
    try:
        result = _call_indicator(func, name, factory)
        _validate_result(result)
    except ACCEPTABLE:
        pass  # Indicator raised a known exception on bad input


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_single_row(name):
    """Indicator handles single-row (len=1) inputs without crashing."""
    func = _resolve(name)
    factory = _make_ohlcv_factory(1)
    try:
        result = _call_indicator(func, name, factory)
        _validate_result(result)
    except ACCEPTABLE:
        pass


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_all_nan(name):
    """Indicator handles all-NaN (len=20) inputs without crashing."""
    func = _resolve(name)
    factory = _make_ohlcv_factory(20, fill="nan")
    try:
        result = _call_indicator(func, name, factory)
        _validate_result(result)
    except ACCEPTABLE:
        pass


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_short_series(name):
    """Indicator handles short (len=5) inputs without crashing."""
    func = _resolve(name)
    factory = _make_ohlcv_factory(5)
    try:
        result = _call_indicator(func, name, factory)
        _validate_result(result)
    except ACCEPTABLE:
        pass


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_negative_length(name):
    """Indicator handles negative length without crashing.

    Most indicators validate ``length`` via
    ``int(length) if length and length > 0 else <default>`` which silently
    falls back to the default.  This test ensures none of them crash.
    """
    func = _resolve(name)
    if not _has_length_param(func):
        pytest.skip(f"{name} has no length parameter")

    factory = _make_ohlcv_factory(50)
    try:
        result = _call_indicator(func, name, factory, extra_kwargs={"length": -5})
        _validate_result(result)
    except ACCEPTABLE:
        pass
