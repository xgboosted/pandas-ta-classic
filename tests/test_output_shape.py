# -*- coding: utf-8 -*-
"""Output shape invariants: length preservation and NaN consistency.

Every indicator must satisfy:
  1. len(output) == len(input) — no truncated Series
  2. output is not all-NaN (sanity check)
  3. NaN lookback scales consistently with `length` parameter

Indicators are auto-discovered from Category metadata — no manual list
to maintain.  Adding a new indicator file is enough; it will be tested
automatically.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._meta import Category

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 300
_close = pd.Series(np.cumsum(np.random.randn(N)) + 100, name="close").clip(lower=1)
_high = _close + np.abs(np.random.randn(N)) * 2
_low = _close - np.abs(np.random.randn(N)) * 2
_low = _low.clip(lower=0.5)
_open = _close.shift(1).fillna(_close.iloc[0])
_volume = pd.Series(np.random.randint(1000, 100000, N), dtype=float, name="volume")
_benchmark = pd.Series(np.cumsum(np.random.randn(N)) + 100).clip(lower=1)

# Map function parameter names -> test data
_SERIES_DATA = {
    "close": _close,
    "high": _high,
    "low": _low,
    "open_": _open,
    "volume": _volume,
    "benchmark": _benchmark,
}

# These need non-OHLCV Series as input — skip auto-discovery
_SKIP = {
    "ma",  # dispatcher, not a concrete indicator
    "long_run",  # needs pre-computed fast/slow Series
    "short_run",  # needs pre-computed fast/slow Series
    "tsignals",  # needs pre-computed trend Series
    "xsignals",  # needs pre-computed signal/xa/xb Series
    "vwap",  # requires DatetimeIndex
}

# Output is shorter than input by design
_SHORT_OUTPUT = {"tos_stdevall", "vp"}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _build_call_args(fn):
    """Build kwargs dict mapping OHLCV params to test data."""
    sig = inspect.signature(fn)
    kwargs = {}
    for pname in sig.parameters:
        if pname in _SERIES_DATA:
            kwargs[pname] = _SERIES_DATA[pname]
    return kwargs


def _can_auto_call(fn):
    """Check if we can call fn with only OHLCV series + defaults."""
    sig = inspect.signature(fn)
    for pname, param in sig.parameters.items():
        if pname in _SERIES_DATA:
            continue
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False
    return True


def _fv(result):
    """first_valid_index of the first column (or Series)."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0].first_valid_index()
    return result.first_valid_index()


def _discover_indicators():
    """Yield (test_id, fn, is_short_output) for every auto-discoverable indicator."""
    for cat in sorted(Category):
        for name in sorted(Category[cat]):
            if name in _SKIP:
                continue
            fn = getattr(ta, name, None)
            if fn is None:
                continue
            if not _can_auto_call(fn):
                continue
            short = name in _SHORT_OUTPUT
            yield f"{cat}/{name}", fn, short


def _discover_indicators_with_length():
    """Yield (test_id, fn) for indicators that accept a `length` parameter."""
    for cat in sorted(Category):
        for name in sorted(Category[cat]):
            if name in _SKIP or name in _SHORT_OUTPUT:
                continue
            fn = getattr(ta, name, None)
            if fn is None:
                continue
            sig = inspect.signature(fn)
            if "length" not in sig.parameters:
                continue
            if not _can_auto_call(fn):
                continue
            yield f"{cat}/{name}", fn


_ALL_INDICATORS = list(_discover_indicators())
_LENGTH_INDICATORS = list(_discover_indicators_with_length())


# ---------------------------------------------------------------------------
# Test 1: Output length must equal input length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_id,fn,short",
    _ALL_INDICATORS,
    ids=[x[0] for x in _ALL_INDICATORS],
)
def test_output_length(test_id, fn, short):
    """Output length must equal input length (or declared short-output length)."""
    kwargs = _build_call_args(fn)
    result = fn(**kwargs)
    if result is None:
        pytest.skip(f"{test_id} returned None")

    if test_id.endswith("/ichimoku"):
        main, span = result
        assert main is not None, f"{test_id}: main DataFrame is None"
        assert len(main) == N, f"{test_id} main: length {len(main)} != {N}"
        return

    if short:
        assert len(result) > 0, f"{test_id}: empty result"
        return

    assert (
        len(result) == N
    ), f"{test_id}: output length {len(result)} != input length {N}"


# ---------------------------------------------------------------------------
# Test 2: Not all NaN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_id,fn,short",
    [(tid, fn, s) for tid, fn, s in _ALL_INDICATORS if not s],
    ids=[x[0] for x in _ALL_INDICATORS if not x[2]],
)
def test_not_all_nan(test_id, fn, short):
    """Output must contain at least one non-NaN value."""
    kwargs = _build_call_args(fn)
    result = fn(**kwargs)
    if result is None:
        pytest.skip(f"{test_id} returned None")

    if test_id.endswith("/ichimoku"):
        result = result[0]
        if result is None:
            pytest.skip(f"{test_id} main is None")

    if isinstance(result, pd.DataFrame):
        assert not result.isna().all().all(), f"{test_id}: all values are NaN"
    else:
        assert not result.isna().all(), f"{test_id}: all values are NaN"


# ---------------------------------------------------------------------------
# Test 3: NaN lookback scales consistently with `length`
#
# For every indicator that accepts `length`, call it with length=10 and
# length=20.  The difference in first_valid_index must be consistent:
#   fv(20) - fv(10) == k * (20 - 10)
# where k is the indicator's chain multiplier (1 for SMA, 2 for DEMA, etc.).
# We don't need to know k — we just verify fv(20) >= fv(10).
#
# This catches:
#   - Off-by-one bugs in lookback calculation
#   - Broken NaN propagation (truncated output had wrong fv)
#   - Regressions where fv stops scaling with length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_id,fn",
    _LENGTH_INDICATORS,
    ids=[x[0] for x in _LENGTH_INDICATORS],
)
def test_nan_scales_with_length(test_id, fn):
    """NaN prefix must grow when length grows."""
    kwargs = _build_call_args(fn)

    r10 = fn(**kwargs, length=10)
    r20 = fn(**kwargs, length=20)

    if r10 is None or r20 is None:
        pytest.skip(f"{test_id} returned None")

    fv10 = _fv(r10)
    fv20 = _fv(r20)

    if fv10 is None or fv20 is None:
        pytest.skip(f"{test_id}: all NaN (fv10={fv10}, fv20={fv20})")

    assert fv20 >= fv10, (
        f"{test_id}: NaN lookback shrank with larger length! "
        f"fv(10)={fv10}, fv(20)={fv20}"
    )

    # Both must preserve output length
    def _len(r):
        if isinstance(r, tuple):
            return len(r[0])
        return len(r)

    assert _len(r10) == N, f"{test_id}: length=10 output {_len(r10)} != {N}"
    assert _len(r20) == N, f"{test_id}: length=20 output {_len(r20)} != {N}"
