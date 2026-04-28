"""
Priority 2 — NaN / warmup behaviour tests.

Three test classes:

TestWarmupNanPrefix
    For window-based indicators run on a sufficiently long synthetic series:
    - The first `warmup` rows are all NaN.
    - Every row AFTER the warmup period is non-NaN (no holes in the output).

TestTooShortInput
    When the input series has fewer rows than the indicator's minimum requirement
    the indicator must return None (not raise, not return all-NaN).

TestBoundaryLength
    When the input series has exactly the minimum required rows there must be
    exactly one valid (non-NaN) value.

Warmup conventions (confirmed empirically against native implementations):
    * length-based rolling (sma, ema, wma, stdev, zscore, rsi, atr, cci, willr):
        warmup = length - 1
    * difference/rate-based (roc, mom):
        warmup = length
    * first-bar-needs-prev-bar (true_range):
        warmup = 1
    * cumulative (obv, ad):
        warmup = 0
"""

from unittest import TestCase

import numpy as np
import pandas as pd

import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Shared synthetic OHLCV data — 200 rows, fully controlled seed
# ---------------------------------------------------------------------------

_N = 200
_RNG = np.random.default_rng(seed=7)
_C = pd.Series(100.0 + np.cumsum(_RNG.standard_normal(_N)))
_H = _C + _RNG.uniform(0.1, 1.0, _N)
_L = _C - _RNG.uniform(0.1, 1.0, _N)
_V = pd.Series(_RNG.integers(1_000_000, 5_000_000, _N).astype(float))
_O = _C + _RNG.standard_normal(_N) * 0.5


def _nan_prefix(result) -> int:
    """Return the number of leading NaN rows in *result* (Series or DataFrame)."""
    if isinstance(result, pd.DataFrame):
        # All columns must agree; use the first one
        series = result.iloc[:, 0]
    else:
        series = result
    # Count consecutive leading NaNs
    non_nan_idx = series.first_valid_index()
    if non_nan_idx is None:
        return len(series)
    return series.index.get_loc(non_nan_idx)


def _no_holes_after_warmup(result, warmup: int) -> bool:
    """Return True when every row at index >= warmup is non-NaN."""
    if isinstance(result, pd.DataFrame):
        tail = result.iloc[warmup:]
        return bool(tail.notna().all(axis=None))
    return bool(result.iloc[warmup:].notna().all())


# ---------------------------------------------------------------------------
# 1 — Warmup NaN prefix
# ---------------------------------------------------------------------------


class TestWarmupNanPrefix(TestCase):
    """Each test asserts the expected NaN prefix length AND no subsequent holes."""

    # --- length-1 warmup (rolling window, first value at index length-1) ---

    def _assert_warmup(self, result, expected_warmup: int, label: str):
        self.assertIsNotNone(
            result, f"{label}: indicator returned None on sufficient data"
        )
        actual = _nan_prefix(result)
        self.assertEqual(
            actual,
            expected_warmup,
            f"{label}: expected {expected_warmup} leading NaN rows, got {actual}",
        )
        self.assertTrue(
            _no_holes_after_warmup(result, expected_warmup),
            f"{label}: NaN gaps found after warmup period",
        )

    def test_sma_warmup(self):
        length = 10
        self._assert_warmup(ta.sma(_C, length), length - 1, "sma")

    def test_ema_warmup(self):
        length = 10
        self._assert_warmup(ta.ema(_C, length, talib=False), length - 1, "ema")

    def test_wma_warmup(self):
        length = 10
        self._assert_warmup(ta.wma(_C, length, talib=False), length - 1, "wma")

    def test_stdev_warmup(self):
        length = 10
        self._assert_warmup(ta.stdev(_C, length, talib=False), length - 1, "stdev")

    def test_zscore_warmup(self):
        length = 10
        self._assert_warmup(ta.zscore(_C, length), length - 1, "zscore")

    def test_rsi_warmup(self):
        length = 14
        self._assert_warmup(ta.rsi(_C, length, talib=False), length - 1, "rsi")

    def test_atr_warmup(self):
        length = 14
        self._assert_warmup(ta.atr(_H, _L, _C, length, talib=False), length - 1, "atr")

    def test_cci_warmup(self):
        length = 14
        self._assert_warmup(ta.cci(_H, _L, _C, length, talib=False), length - 1, "cci")

    def test_willr_warmup(self):
        length = 14
        self._assert_warmup(
            ta.willr(_H, _L, _C, length, talib=False), length - 1, "willr"
        )

    # --- length warmup (diff / rate: first valid value at index = length) ---

    def test_roc_warmup(self):
        length = 10
        self._assert_warmup(ta.roc(_C, length, talib=False), length, "roc")

    def test_mom_warmup(self):
        length = 10
        self._assert_warmup(ta.mom(_C, length, talib=False), length, "mom")

    # --- 1-row warmup ---

    def test_true_range_warmup(self):
        self._assert_warmup(ta.true_range(_H, _L, _C, talib=False), 1, "true_range")

    # --- zero warmup (cumulative, starts at row 0) ---

    def test_obv_no_warmup(self):
        result = ta.obv(_C, _V, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(int(result.isna().sum()), 0, "obv should have 0 NaN rows")

    def test_ad_no_warmup(self):
        result = ta.ad(_H, _L, _C, _V, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(int(result.isna().sum()), 0, "ad should have 0 NaN rows")


# ---------------------------------------------------------------------------
# 2 — Too-short input: must return None
# ---------------------------------------------------------------------------


class TestTooShortInput(TestCase):
    """Indicators must return None (not raise) when input is shorter than required."""

    def _short(self, n: int):
        c = _C.iloc[:n]
        h = _H.iloc[:n]
        l = _L.iloc[:n]
        v = _V.iloc[:n]
        return c, h, l, v

    def test_sma_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.sma(c, length=20), "sma should return None when len < length"
        )

    def test_ema_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.ema(c, length=20, talib=False),
            "ema should return None when len < length",
        )

    def test_rsi_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.rsi(c, length=14, talib=False),
            "rsi should return None when len < length",
        )

    def test_atr_too_short(self):
        c, h, l, _ = self._short(5)
        self.assertIsNone(
            ta.atr(h, l, c, length=14, talib=False),
            "atr should return None when len < length",
        )

    def test_roc_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.roc(c, length=10, talib=False),
            "roc should return None when len < length",
        )

    def test_stdev_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.stdev(c, length=20, talib=False),
            "stdev should return None when len < length",
        )

    def test_bbands_too_short(self):
        c, *_ = self._short(5)
        self.assertIsNone(
            ta.bbands(c, length=20, talib=False),
            "bbands should return None when len < length",
        )

    def test_adx_too_short(self):
        c, h, l, _ = self._short(5)
        self.assertIsNone(
            ta.adx(h, l, c, length=14, talib=False),
            "adx should return None when len < length",
        )


# ---------------------------------------------------------------------------
# 3 — Boundary: exactly minimum rows → exactly one valid value
# ---------------------------------------------------------------------------


class TestBoundaryLength(TestCase):
    """When series length == required minimum, exactly one row must be non-NaN."""

    def _exact(self, n: int):
        c = _C.iloc[:n]
        h = _H.iloc[:n]
        l = _L.iloc[:n]
        return c, h, l

    def test_sma_boundary(self):
        length = 20
        c, *_ = self._exact(length)
        result = ta.sma(c, length=length)
        self.assertIsNotNone(result, "sma should not return None at exact boundary")
        self.assertEqual(
            int(result.notna().sum()),
            1,
            f"sma({length}) on {length} rows should have exactly 1 valid value",
        )

    def test_ema_boundary(self):
        length = 10
        c, *_ = self._exact(length)
        result = ta.ema(c, length=length, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(
            int(result.notna().sum()),
            1,
            f"ema({length}) on {length} rows should have exactly 1 valid value",
        )

    def test_rsi_boundary(self):
        # RSI needs length rows to produce 1 valid output (warmup = length-1)
        length = 14
        c, *_ = self._exact(length)
        result = ta.rsi(c, length=length, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(
            int(result.notna().sum()),
            1,
            f"rsi({length}) on {length} rows should have exactly 1 valid value",
        )

    def test_roc_boundary(self):
        # ROC needs length+1 rows for the first valid output (warmup = length)
        length = 10
        c, *_ = self._exact(length + 1)
        result = ta.roc(c, length=length, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(
            int(result.notna().sum()),
            1,
            f"roc({length}) on {length+1} rows should have exactly 1 valid value",
        )

    def test_stdev_boundary(self):
        length = 20
        c, *_ = self._exact(length)
        result = ta.stdev(c, length=length, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(
            int(result.notna().sum()),
            1,
            f"stdev({length}) on {length} rows should have exactly 1 valid value",
        )
