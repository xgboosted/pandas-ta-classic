# -*- coding: utf-8 -*-
"""Output shape invariants: length preservation and NaN consistency.

Every indicator must satisfy:
  1. len(output) == len(input) — no truncated Series
  2. output is not all-NaN (sanity check)
  3. NaN lookback scales consistently with `length` parameter
  4. talib=True and talib=False produce same length and same first_valid_index

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

try:
    import talib as _tal

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

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


def _discover_talib_indicators():
    """Yield (test_id, fn) for indicators that accept a `talib` parameter."""
    for cat in sorted(Category):
        for name in sorted(Category[cat]):
            if name in _SKIP or name in _SHORT_OUTPUT:
                continue
            fn = getattr(ta, name, None)
            if fn is None:
                continue
            sig = inspect.signature(fn)
            if "talib" not in sig.parameters:
                continue
            if not _can_auto_call(fn):
                continue
            yield f"{cat}/{name}", fn


_ALL_INDICATORS = list(_discover_indicators())
_LENGTH_INDICATORS = list(_discover_indicators_with_length())
_TALIB_INDICATORS = list(_discover_talib_indicators())

# Known first_valid_index mismatches between native and TA-Lib paths.
# Tracked here so they show up as xfail (not silently hidden).
_TALIB_FV_XFAIL = {
    "momentum/cmo": "RMA off-by-one: native fv=13, talib fv=14",
    "momentum/rsi": "RMA off-by-one: native fv=13, talib fv=14",
    "momentum/uo": "off-by-one: native fv=27, talib fv=28",
    "momentum/macd": "Values exact; native outputs MACD line from bar 25, talib from bar 33",
    "volatility/atr": "RMA off-by-one: native fv=13, talib fv=14",
    "volume/mfi": "RMA off-by-one: native fv=13, talib fv=14",
    "cycles/ht_dcperiod": "Hilbert warmup: native fv=0, talib fv=32",
    "cycles/ht_dcphase": "Hilbert warmup: native fv=37, talib fv=63",
    "cycles/ht_phasor": "Hilbert warmup: native fv=12, talib fv=32",
    "cycles/ht_sine": "Hilbert warmup: native fv=37, talib fv=63",
    "overlap/ht_trendline": "Hilbert warmup: native fv=37, talib fv=63",
}


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


# ---------------------------------------------------------------------------
# Test 4: talib=True vs talib=False consistency
#
# For every indicator with a `talib` parameter, verify that both paths
# produce the same output length and the same first_valid_index.
# Known mismatches are tracked as xfail so they don't block CI but remain
# visible.  When a mismatch is fixed the xfail will xpass and pytest will
# flag it for removal.
# ---------------------------------------------------------------------------


def _make_talib_params():
    """Build parametrize list with xfail markers for known mismatches."""
    params = []
    ids = []
    for test_id, fn in _TALIB_INDICATORS:
        if test_id in _TALIB_FV_XFAIL:
            params.append(
                pytest.param(
                    test_id,
                    fn,
                    marks=pytest.mark.xfail(reason=_TALIB_FV_XFAIL[test_id]),
                )
            )
        else:
            params.append((test_id, fn))
        ids.append(test_id)
    return params, ids


_TALIB_PARAMS, _TALIB_IDS = _make_talib_params()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
@pytest.mark.parametrize("test_id,fn", _TALIB_PARAMS, ids=_TALIB_IDS)
def test_talib_consistency(test_id, fn):
    """talib=True and talib=False must produce same length and NaN lookback."""
    kwargs = _build_call_args(fn)

    r_native = fn(**kwargs, talib=False)
    r_talib = fn(**kwargs, talib=True)

    if r_native is None or r_talib is None:
        pytest.skip(f"{test_id} returned None")

    def _result_len(r):
        if isinstance(r, tuple):
            return len(r[0])
        return len(r)

    len_nat = _result_len(r_native)
    len_tal = _result_len(r_talib)
    assert len_nat == N, f"{test_id}: native length {len_nat} != {N}"
    assert len_tal == N, f"{test_id}: talib length {len_tal} != {N}"

    fv_nat = _fv(r_native)
    fv_tal = _fv(r_talib)
    assert fv_nat == fv_tal, (
        f"{test_id}: first_valid_index mismatch — " f"native={fv_nat}, talib={fv_tal}"
    )


# ---------------------------------------------------------------------------
# Test 5: No data leak (future-data / lookahead bias detection)
#
# For each indicator, compute on full data (N rows) and on truncated data
# (first M rows).  The first M rows of both results must be identical.
# If an indicator peeks at future data (e.g. .shift(-k)), the truncated
# result will differ → test fails → data leak detected.
#
# Two indicators intentionally support lookahead via a `lookahead` kwarg
# (default True): dpo and ichimoku.  These are marked xfail here and
# verified separately in Test 6.
# ---------------------------------------------------------------------------

_LEAK_CUT = 50  # rows removed from the end for truncation test

_LOOKAHEAD_XFAIL = {
    "trend/dpo": "centered=True uses .shift(-t) lookahead",
    "overlap/ichimoku": "chikou_span uses close.shift(-kijun) lookahead",
}


def _discover_leak_test_indicators():
    """Yield (test_id, fn) for data-leak testing."""
    for test_id, fn, short in _ALL_INDICATORS:
        if short:
            continue
        yield test_id, fn


def _make_leak_params():
    params, ids = [], []
    for test_id, fn in _discover_leak_test_indicators():
        if test_id in _LOOKAHEAD_XFAIL:
            params.append(
                pytest.param(
                    test_id,
                    fn,
                    marks=pytest.mark.xfail(reason=_LOOKAHEAD_XFAIL[test_id]),
                )
            )
        else:
            params.append((test_id, fn))
        ids.append(test_id)
    return params, ids


_LEAK_PARAMS, _LEAK_IDS = _make_leak_params()


@pytest.mark.parametrize("test_id,fn", _LEAK_PARAMS, ids=_LEAK_IDS)
def test_no_data_leak(test_id, fn):
    """Indicator output must not change when future data is removed."""
    M = N - _LEAK_CUT

    kwargs_full = _build_call_args(fn)
    kwargs_trunc = {k: v.iloc[:M].copy() for k, v in kwargs_full.items()}

    result_full = fn(**kwargs_full)
    result_trunc = fn(**kwargs_trunc)

    if result_full is None or result_trunc is None:
        pytest.skip(f"{test_id} returned None")

    # Unpack tuples (ichimoku)
    if isinstance(result_full, tuple):
        result_full = result_full[0]
    if isinstance(result_trunc, tuple):
        result_trunc = result_trunc[0]

    full_head = result_full.iloc[:M]

    if isinstance(full_head, pd.DataFrame):
        common = full_head.columns.intersection(result_trunc.columns)
        pd.testing.assert_frame_equal(
            full_head[common].reset_index(drop=True),
            result_trunc[common].reset_index(drop=True),
            check_names=False,
        )
    else:
        pd.testing.assert_series_equal(
            full_head.reset_index(drop=True),
            result_trunc.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Test 6: lookahead=False eliminates data leak
#
# dpo and ichimoku support lookahead=False to disable their forward shift.
# Verify that this actually fixes the data leak.
# ---------------------------------------------------------------------------

_LOOKAHEAD_FIX_PARAMS = [
    ("trend/dpo", ta.dpo),
    ("overlap/ichimoku", ta.ichimoku),
]


@pytest.mark.parametrize(
    "test_id,fn",
    _LOOKAHEAD_FIX_PARAMS,
    ids=[x[0] for x in _LOOKAHEAD_FIX_PARAMS],
)
def test_lookahead_false_fixes_leak(test_id, fn):
    """lookahead=False must eliminate future data dependency."""
    M = N - _LEAK_CUT

    kwargs_full = _build_call_args(fn)
    kwargs_trunc = {k: v.iloc[:M].copy() for k, v in kwargs_full.items()}

    result_full = fn(**kwargs_full, lookahead=False)
    result_trunc = fn(**kwargs_trunc, lookahead=False)

    # Unpack tuples (ichimoku)
    if isinstance(result_full, tuple):
        result_full = result_full[0]
    if isinstance(result_trunc, tuple):
        result_trunc = result_trunc[0]

    full_head = result_full.iloc[:M]

    if isinstance(full_head, pd.DataFrame):
        pd.testing.assert_frame_equal(
            full_head.reset_index(drop=True),
            result_trunc.reset_index(drop=True),
            check_names=False,
        )
    else:
        pd.testing.assert_series_equal(
            full_head.reset_index(drop=True),
            result_trunc.reset_index(drop=True),
            check_names=False,
        )
