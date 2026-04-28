"""
Priority 3 — Edge-case / boundary input tests.

Four test classes, each probing a different class of abnormal input:

TestAllNanInput
    All-NaN series must produce a result (not None, no exception) that is
    itself all-NaN — no crashes, no silent wrong values.

TestConstantInput
    Constant price series exercises degenerate arithmetic:
    - sma returns the constant value for all valid rows
    - stdev returns 0 for all valid rows
    - rsi returns all-NaN (no gains or losses → division by zero in RSI formula)

TestInfInInput
    A single ±Inf value inserted mid-series must not crash any indicator.
    The indicator may propagate Inf or NaN, but must return a non-None result
    of the correct type.

TestMismatchedLengths
    When one OHLCV component is shorter than required (below minimum threshold)
    the indicator must return None rather than raise.
    When one component is shorter than the others but still above the threshold,
    the valid output rows should equal the length of the shorter component.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Shared synthetic series
# ---------------------------------------------------------------------------

_N = 60
_RNG = np.random.default_rng(seed=99)
_C = pd.Series(100.0 + np.cumsum(_RNG.standard_normal(_N)))
_H = _C + _RNG.uniform(0.2, 1.0, _N)
_L = _C - _RNG.uniform(0.2, 1.0, _N)
_V = pd.Series(_RNG.integers(500_000, 3_000_000, _N).astype(float))


def _all_nan_series(n: int = _N) -> pd.Series:
    return pd.Series([float("nan")] * n)


# ---------------------------------------------------------------------------
# 1 — All-NaN input
# ---------------------------------------------------------------------------


class TestAllNanInput(TestCase):
    """Indicators on all-NaN input must return an all-NaN result, not raise."""

    def _assert_all_nan_result(self, result, label: str):
        self.assertIsNotNone(
            result, f"{label}: returned None instead of all-NaN result"
        )
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                self.assertTrue(
                    result[col].isna().all(),
                    f"{label}[{col}]: expected all-NaN column, got non-NaN values",
                )
        else:
            self.assertTrue(result.isna().all(), f"{label}: expected all-NaN Series")

    def test_sma_all_nan(self):
        self._assert_all_nan_result(ta.sma(_all_nan_series(), 10), "sma")

    def test_ema_all_nan(self):
        self._assert_all_nan_result(ta.ema(_all_nan_series(), 10, talib=False), "ema")

    def test_rsi_all_nan(self):
        self._assert_all_nan_result(ta.rsi(_all_nan_series(), 14, talib=False), "rsi")

    def test_atr_all_nan(self):
        nan = _all_nan_series()
        self._assert_all_nan_result(ta.atr(nan, nan, nan, 14, talib=False), "atr")

    def test_obv_all_nan(self):
        nan = _all_nan_series()
        self._assert_all_nan_result(ta.obv(nan, nan, talib=False), "obv")

    def test_stdev_all_nan(self):
        self._assert_all_nan_result(
            ta.stdev(_all_nan_series(), 10, talib=False), "stdev"
        )

    def test_roc_all_nan(self):
        self._assert_all_nan_result(ta.roc(_all_nan_series(), 10, talib=False), "roc")


# ---------------------------------------------------------------------------
# 2 — Constant input (degenerate arithmetic)
# ---------------------------------------------------------------------------

_CONSTANT = 100.0
_CONST_C = pd.Series([_CONSTANT] * _N)
_CONST_H = pd.Series([_CONSTANT + 1.0] * _N)
_CONST_L = pd.Series([_CONSTANT - 1.0] * _N)


class TestConstantInput(TestCase):
    """Constant price series should not crash and should return meaningful values."""

    def test_sma_constant_equals_value(self):
        """SMA of a constant series must equal the constant for all valid rows."""
        result = ta.sma(_CONST_C, length=10)
        self.assertIsNotNone(result)
        valid = result.dropna()
        self.assertFalse(valid.empty, "sma on constant returned all-NaN")
        self.assertTrue(
            (valid == _CONSTANT).all(),
            f"sma(constant={_CONSTANT}): not all valid values equal the constant",
        )

    def test_stdev_constant_is_zero(self):
        """Standard deviation of a constant series must be 0 for all valid rows."""
        result = ta.stdev(_CONST_C, length=10, talib=False)
        self.assertIsNotNone(result)
        valid = result.dropna()
        self.assertFalse(valid.empty, "stdev on constant returned all-NaN")
        self.assertTrue(
            (valid.abs() < 1e-10).all(), "stdev of constant series must be 0"
        )

    def test_rsi_constant_is_all_nan(self):
        """RSI on a constant series is undefined (0/0) — result must be all-NaN."""
        result = ta.rsi(_CONST_C, length=14, talib=False)
        self.assertIsNotNone(result)
        self.assertTrue(
            result.isna().all(),
            "rsi of constant series should be all-NaN (no gains or losses)",
        )

    def test_atr_constant_returns_valid(self):
        """ATR on constant H/L/C should not crash and must produce finite values."""
        result = ta.atr(_CONST_H, _CONST_L, _CONST_C, length=14, talib=False)
        self.assertIsNotNone(result)
        valid = result.dropna()
        self.assertFalse(valid.empty, "atr on constant returned all-NaN")
        self.assertTrue(
            np.isfinite(valid.values).all(),
            "atr on constant series produced non-finite values",
        )

    def test_bbands_constant_no_crash(self):
        """Bollinger Bands on constant input must not crash and must return a DataFrame."""
        result = ta.bbands(_CONST_C, length=10, talib=False)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_mom_constant_is_zero(self):
        """Momentum (close[i] - close[i-length]) of a constant series is always 0."""
        result = ta.mom(_CONST_C, length=10, talib=False)
        self.assertIsNotNone(result)
        valid = result.dropna()
        self.assertFalse(valid.empty, "mom on constant returned all-NaN")
        self.assertTrue((valid.abs() < 1e-10).all(), "mom of constant series must be 0")

    def test_roc_constant_is_zero(self):
        """Rate of change on a constant series is always 0 %."""
        result = ta.roc(_CONST_C, length=10, talib=False)
        self.assertIsNotNone(result)
        valid = result.dropna()
        self.assertFalse(valid.empty, "roc on constant returned all-NaN")
        self.assertTrue((valid.abs() < 1e-10).all(), "roc of constant series must be 0")


# ---------------------------------------------------------------------------
# 3 — Inf values in input
# ---------------------------------------------------------------------------


class TestInfInInput(TestCase):
    """A single Inf in input must not raise; indicator must return a non-None result."""

    def _inject_inf(self, series: pd.Series, pos: int = 25) -> pd.Series:
        s = series.copy()
        s.iloc[pos] = float("inf")
        return s

    def _assert_no_crash(self, result, label: str):
        self.assertIsNotNone(result, f"{label}: returned None when Inf was in input")
        # Just verify type — we don't mandate how Inf propagates
        self.assertIsInstance(
            result,
            (pd.Series, pd.DataFrame),
            f"{label}: unexpected return type {type(result)}",
        )

    def test_sma_inf_input(self):
        self._assert_no_crash(ta.sma(self._inject_inf(_C), 10), "sma")

    def test_ema_inf_input(self):
        self._assert_no_crash(ta.ema(self._inject_inf(_C), 10, talib=False), "ema")

    def test_rsi_inf_input(self):
        self._assert_no_crash(ta.rsi(self._inject_inf(_C), 14, talib=False), "rsi")

    def test_stdev_inf_input(self):
        self._assert_no_crash(ta.stdev(self._inject_inf(_C), 10, talib=False), "stdev")

    def test_roc_inf_input(self):
        self._assert_no_crash(ta.roc(self._inject_inf(_C), 10, talib=False), "roc")

    def test_obv_inf_volume(self):
        """Inf in volume column must not crash OBV."""
        self._assert_no_crash(
            ta.obv(_C, self._inject_inf(_V), talib=False), "obv(inf_volume)"
        )


# ---------------------------------------------------------------------------
# 4 — Mismatched input lengths
# ---------------------------------------------------------------------------


class TestMismatchedLengths(TestCase):
    """Shorter OHLCV component below the minimum threshold → None (no crash)."""

    def test_atr_h_too_short_returns_none(self):
        """atr with h shorter than the minimum requirement must return None."""
        result = ta.atr(_H.iloc[:5], _L, _C, length=14, talib=False)
        self.assertIsNone(
            result, "atr should return None when h is shorter than required"
        )

    def test_atr_l_too_short_returns_none(self):
        result = ta.atr(_H, _L.iloc[:5], _C, length=14, talib=False)
        self.assertIsNone(
            result, "atr should return None when l is shorter than required"
        )

    def test_atr_c_too_short_returns_none(self):
        result = ta.atr(_H, _L, _C.iloc[:5], length=14, talib=False)
        self.assertIsNone(
            result, "atr should return None when c is shorter than required"
        )

    def test_adx_h_too_short_returns_none(self):
        result = ta.adx(_H.iloc[:5], _L, _C, length=14, talib=False)
        self.assertIsNone(
            result, "adx should return None when h is shorter than required"
        )

    def test_obv_short_volume_limits_output(self):
        """
        When volume is shorter than close but still produces valid rows,
        the non-NaN count should equal len(volume).
        """
        v_short = _V.iloc[:20]
        result = ta.obv(_C, v_short, talib=False)
        self.assertIsNotNone(result)
        non_nan = int(result.notna().sum())
        self.assertEqual(
            non_nan,
            len(v_short),
            f"obv with short volume: expected {len(v_short)} non-NaN rows, got {non_nan}",
        )
