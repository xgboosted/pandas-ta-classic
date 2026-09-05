"""Issue #43 — Regression tests for bug fixes and critical features.

Pins the correct post-fix behaviour for changes documented in CHANGELOG.md
so that regressions are immediately detected.

Covered fixes:
  1. stdev / variance  — ddof defaults to 0 (population), not 1 (sample)
  2. linreg            — degrees kwarg defaults to True, so angle=True returns
                         values in degrees, not radians
  3. QQE               — returns >= 6 columns including 3 new band/direction
                         columns: QQEb_l, QQEb_s, QQEd
  4. emv               — divisor defaults to 10000 (tulipy convention)
  5. zscore            — column name is ZS_{length} (was Z_{length})
  6. rvgi              — returns 3 columns including histogram (RVGIh_*)
  7. hl2 / hlc3        — return None on invalid input; respect fillna kwarg
  8. cdl_z             — full=True path uses bfill() (pandas 3.0 compatible)
  9. edecay            — multiplicative decay floored at close (not additive)
 10. psl               — open_ branch returns None when verify_series returns None
 11. apply_fill        — fillna kwarg is honoured by hl2, hlc3, avgprice, emv, edecay
 12. is_datetime_ordered — returns bool (no return-in-finally SyntaxWarning)
 13. ad                — invalid optional open_ returns None instead of crashing
 14. ad                — missing required-arg None guard added
 15. cmf               — invalid optional open_ returns None instead of crashing
 16. psar              — invalid optional close returns None instead of crashing
 17. dema/tema/t3/trima/cci/natr — talib=False propagated to sub-indicator calls
 18. dm                — short input returns None (min_length was missing), matching
                         plus_dm/minus_dm and the documented short-input contract
 19. cdl_pattern       — short input no longer raises AttributeError; sub-patterns
                         that return None are skipped instead of dereferenced
 20. ema               — the SMA seed no longer indexes past the end when fewer
                         than `length` valid values follow the first valid one;
                         fixes IndexError in 14 chained indicators (trix, tsi,
                         qqe, ppo, pvo, ...) on clean data with default args
 21. ht_* (_hilbert)   — a single NaN in the input propagates as NaN instead of
                         raising ValueError in int(nan) at the DCPeriod rounding
 22. candle_color      — a NaN open/close yields NaN instead of raising
                         IntCastingNaNError; affects cdl_inside and cdl_pattern
 23. rma / linreg /    — short-window hardening: these three sites crashed if the
     _sliding_weighted_   verify_series min_length guard was bypassed. Not
     ma                   reachable through the public API; guarded anyway

Run:
    python -m unittest tests/test_regression_bugfixes.py
"""

import importlib
import inspect
import math
from unittest import TestCase

import numpy as np
import pandas as pd

import pandas_ta_classic as ta
from tests.config import get_sample_data

# ---------------------------------------------------------------------------
# Fix 1: stdev / variance ddof default = 0 (population)
# ---------------------------------------------------------------------------


class TestStdevDdofDefault(TestCase):
    """stdev default ddof=0 (population std, not sample std)."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def test_default_matches_population_stdev(self):
        """ta.stdev() without ddof kwarg must equal pandas rolling std(ddof=0)."""
        length = 30
        result = ta.stdev(self.close, length=length, talib=False)
        expected = self.close.rolling(length).std(ddof=0)

        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any(), "No overlapping non-NaN values to compare")
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="stdev() default ddof must be 0 (population std)",
        )

    def test_default_differs_from_sample_stdev(self):
        """ta.stdev() default must NOT equal pandas rolling std(ddof=1)."""
        length = 30
        result = ta.stdev(self.close, length=length, talib=False)
        sample_std = self.close.rolling(length).std(ddof=1)

        mask = result.notna() & sample_std.notna()
        diff = np.abs(result[mask].values - sample_std[mask].values)
        self.assertTrue(
            diff.max() > 1e-8,
            "stdev() default must NOT equal sample std (ddof=1); " "ddof=0 (population) is the correct post-fix default",
        )

    def test_explicit_ddof1_matches_sample_stdev(self):
        """ta.stdev(ddof=1) must equal pandas rolling std(ddof=1)."""
        length = 30
        result = ta.stdev(self.close, length=length, ddof=1, talib=False)
        expected = self.close.rolling(length).std(ddof=1)

        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any())
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="stdev(ddof=1) must equal sample standard deviation",
        )


# ---------------------------------------------------------------------------
# Fix 2: linreg degrees default = True
# ---------------------------------------------------------------------------


class TestLinregDegreesDefault(TestCase):
    """linreg degrees kwarg defaults to True: angle=True returns degrees."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def test_angle_default_returns_degrees(self):
        """linreg(angle=True) without degrees kwarg must return degree values.

        For SPY_D.csv the maximum absolute angle clearly exceeds pi/2 (1.57)
        when expressed in degrees — which is impossible if the values were
        radians (radians are bounded by ±pi/2 for slope angles).
        """
        length = 14
        result = ta.linreg(self.close, length=length, angle=True, talib=False)
        self.assertIsNotNone(result)
        vals = result.dropna()
        self.assertGreater(len(vals), 0)

        max_abs = float(vals.abs().max())
        self.assertGreater(
            max_abs,
            math.pi / 2,
            f"linreg(angle=True) max|value|={max_abs:.4f} must exceed " "pi/2 ({:.4f}), confirming values are in degrees".format(math.pi / 2),
        )

    def test_explicit_degrees_false_returns_radians(self):
        """linreg(angle=True, degrees=False) must return radian values in (-pi/2, pi/2)."""
        length = 14
        result = ta.linreg(self.close, length=length, angle=True, degrees=False, talib=False)
        self.assertIsNotNone(result)
        vals = result.dropna()
        self.assertGreater(len(vals), 0)

        self.assertTrue(
            (vals.abs() <= (math.pi / 2 + 1e-9)).all(),
            "linreg(angle=True, degrees=False) must return radians in (-pi/2, pi/2)",
        )

    def test_degrees_and_radians_produce_different_values(self):
        """degrees=True and degrees=False must produce numerically different results."""
        length = 14
        deg = ta.linreg(self.close, length=length, angle=True, degrees=True, talib=False)
        rad = ta.linreg(self.close, length=length, angle=True, degrees=False, talib=False)
        mask = deg.notna() & rad.notna()
        diff = (deg[mask] - rad[mask]).abs()
        self.assertTrue(
            diff.max() > 1e-6,
            "degrees=True and degrees=False must produce different angle values",
        )


# ---------------------------------------------------------------------------
# Fix 3: QQE column count (was 3, now >= 6 including QQEb_l, QQEb_s, QQEd)
# ---------------------------------------------------------------------------


class TestQqeColumnCount(TestCase):
    """QQE returns >= 6 columns including the 3 new band/direction columns."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def _result(self):
        return ta.qqe(self.close)

    def test_qqe_returns_dataframe(self):
        self.assertIsInstance(self._result(), pd.DataFrame)

    def test_qqe_column_count_gte_6(self):
        """QQE must return >= 6 columns (was 3 before the fix)."""
        result = self._result()
        self.assertGreaterEqual(
            len(result.columns),
            6,
            f"QQE must return >=6 columns; got {len(result.columns)}: {list(result.columns)}",
        )

    def test_qqe_long_band_column_present(self):
        cols = list(self._result().columns)
        self.assertTrue(
            any("QQEb_l" in c for c in cols),
            f"QQE missing 'QQEb_l' long-band column. Got: {cols}",
        )

    def test_qqe_short_band_column_present(self):
        cols = list(self._result().columns)
        self.assertTrue(
            any("QQEb_s" in c for c in cols),
            f"QQE missing 'QQEb_s' short-band column. Got: {cols}",
        )

    def test_qqe_direction_column_present(self):
        cols = list(self._result().columns)
        self.assertTrue(
            any("QQEd" in c for c in cols),
            f"QQE missing 'QQEd' direction column. Got: {cols}",
        )

    def test_qqe_direction_values_binary(self):
        """QQEd must contain only +1.0 or -1.0 after the warmup period."""
        result = self._result()
        trend_col = next(c for c in result.columns if "QQEd" in c)
        vals = result[trend_col].dropna()
        unique_vals = set(vals.unique())
        self.assertTrue(
            unique_vals.issubset({1.0, -1.0}),
            f"QQEd must contain only {{+1, -1}}, got: {unique_vals}",
        )


# ---------------------------------------------------------------------------
# Fix 4: emv divisor defaults to 10000 (tulipy convention)
# ---------------------------------------------------------------------------


class TestEmvDivisorDefault(TestCase):
    """emv divisor defaults to 10000 (tulipy convention post-fix)."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.volume = cls.df["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_emv_default_uses_divisor_10000(self):
        """ta.emv() must match manual calculation with divisor=10000."""
        h, low, v = self.high, self.low, self.volume
        hl_range = (h - low).replace(0, float("nan"))
        midpoint = 0.5 * (h + low)
        distance = midpoint - midpoint.shift(1)
        box_ratio = (v / 10_000) / hl_range
        expected = distance / box_ratio

        result = ta.emv(h, low, v)
        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any())
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="emv() default divisor must be 10000 (tulipy convention)",
        )

    def test_emv_custom_divisor_differs_from_default(self):
        """emv(divisor=1) must produce a different result than the default."""
        h, low, v = self.high, self.low, self.volume
        default = ta.emv(h, low, v)
        custom = ta.emv(h, low, v, divisor=1)

        mask = default.notna() & custom.notna()
        diff = (default[mask] - custom[mask]).abs()
        self.assertTrue(
            diff.max() > 1e-6,
            "emv divisor=1 must differ from the default divisor=10000",
        )


# ---------------------------------------------------------------------------
# Fix 5: zscore column name changed from Z_{length} to ZS_{length}
# ---------------------------------------------------------------------------


class TestZscoreColumnName(TestCase):
    """zscore output Series name must be ZS_{length}, not Z_{length}."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.close

    def test_zscore_column_name_prefix(self):
        """zscore() result name must start with 'ZS_', not 'Z_'."""
        length = 30
        result = ta.zscore(self.close, length=length)
        self.assertIsNotNone(result)
        self.assertTrue(
            result.name.startswith("ZS_"),
            f"zscore column name must start with 'ZS_'; got '{result.name}'",
        )

    def test_zscore_column_name_exact(self):
        """zscore(length=20) must produce column named 'ZS_20'."""
        result = ta.zscore(self.close, length=20)
        self.assertEqual(result.name, "ZS_20")

    def test_zscore_old_name_not_used(self):
        """zscore must not use old 'Z_{length}' naming convention."""
        result = ta.zscore(self.close, length=14)
        self.assertNotEqual(
            result.name,
            "Z_14",
            "zscore name 'Z_14' is the old pre-fix name; must be 'ZS_14'",
        )


# ---------------------------------------------------------------------------
# Fix 6: rvgi returns 3 columns including the histogram column (RVGIh_*)
# ---------------------------------------------------------------------------


class TestRvgiHistogram(TestCase):
    """rvgi() must return 3 columns: RVGI, Signal, and Histogram."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def _result(self):
        return ta.rvgi(self.df["open"], self.df["high"], self.df["low"], self.df["close"])

    def test_rvgi_returns_dataframe(self):
        self.assertIsInstance(self._result(), pd.DataFrame)

    def test_rvgi_column_count_is_3(self):
        """rvgi must return exactly 3 columns (rvgi, signal, histogram)."""
        result = self._result()
        self.assertEqual(
            len(result.columns),
            3,
            f"rvgi must return 3 columns; got {len(result.columns)}: {list(result.columns)}",
        )

    def test_rvgi_histogram_column_present(self):
        """rvgi result must contain a histogram column named RVGIh_*."""
        result = self._result()
        hist_cols = [c for c in result.columns if c.startswith("RVGIh")]
        self.assertTrue(
            len(hist_cols) == 1,
            f"rvgi must contain exactly one 'RVGIh_*' column; got: {list(result.columns)}",
        )

    def test_rvgi_histogram_equals_rvgi_minus_signal(self):
        """Histogram column must equal RVGI minus Signal."""
        result = self._result()
        rvgi_col = next(c for c in result.columns if c.startswith("RVGI_"))
        sig_col = next(c for c in result.columns if c.startswith("RVGIs"))
        hist_col = next(c for c in result.columns if c.startswith("RVGIh"))
        mask = result[rvgi_col].notna() & result[sig_col].notna() & result[hist_col].notna()
        np.testing.assert_allclose(
            result.loc[mask, hist_col].values,
            (result.loc[mask, rvgi_col] - result.loc[mask, sig_col]).values,
            rtol=1e-6,
            err_msg="RVGIh must equal RVGI - RVGIs",
        )


# ---------------------------------------------------------------------------
# Fix 7: hl2 / hlc3 None guard on invalid series input
# ---------------------------------------------------------------------------


class TestHl2Hlc3NoneGuard(TestCase):
    """hl2 and hlc3 must return None when given an invalid/empty series."""

    def test_hl2_returns_none_on_none_high(self):
        """hl2(None, valid) must return None, not raise."""
        result = ta.hl2(None, pd.Series([1.0, 2.0], dtype=float))
        self.assertIsNone(result)

    def test_hlc3_returns_none_on_none_high(self):
        """hlc3(None, valid, valid) must return None, not raise."""
        v = pd.Series([1.0, 2.0], dtype=float)
        result = ta.hlc3(None, v, v, talib=False)
        self.assertIsNone(result)

    def test_hl2_valid_input_returns_series(self):
        """hl2 with valid inputs must return a non-empty Series named 'HL2'."""
        h = pd.Series([10.0, 11.0, 12.0])
        low = pd.Series([8.0, 9.0, 10.0])
        result = ta.hl2(h, low)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "HL2")
        np.testing.assert_allclose(result.values, [9.0, 10.0, 11.0], rtol=1e-9)

    def test_hlc3_valid_input_returns_series(self):
        """hlc3 with valid inputs must return a Series named 'HLC3'."""
        h = pd.Series([12.0, 13.0])
        low = pd.Series([8.0, 9.0])
        c = pd.Series([10.0, 11.0])
        result = ta.hlc3(h, low, c, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "HLC3")
        np.testing.assert_allclose(result.values, [10.0, 11.0], rtol=1e-9)


# ---------------------------------------------------------------------------
# Fix 8: cdl_z full=True uses bfill() — pandas 3.0 compatible
# ---------------------------------------------------------------------------


class TestCdlZBfill(TestCase):
    """cdl_z(full=True) must back-fill early NaN values (bfill fix)."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_cdlz_full_true_no_leading_nans(self):
        """cdl_z(full=True) result must have no NaN in the open_Z column."""
        result = ta.cdl_z(
            self.df["open"],
            self.df["high"],
            self.df["low"],
            self.df["close"],
            full=True,
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        open_col = next(c for c in result.columns if "open" in c.lower())
        nan_count = result[open_col].isna().sum()
        self.assertEqual(
            nan_count,
            0,
            f"cdl_z(full=True) must bfill all NaNs; found {nan_count} NaN values",
        )

    def test_cdlz_full_false_has_leading_nans(self):
        """cdl_z(full=False) must have leading NaN values (warmup period)."""
        length = 30
        result = ta.cdl_z(
            self.df["open"],
            self.df["high"],
            self.df["low"],
            self.df["close"],
            full=False,
            length=length,
        )
        self.assertIsNotNone(result)
        open_col = next(c for c in result.columns if "open" in c.lower())
        nan_count = result[open_col].isna().sum()
        self.assertGreater(
            nan_count,
            0,
            "cdl_z(full=False) must have leading NaN values in warmup period",
        )


# ---------------------------------------------------------------------------
# Fix 9: edecay is multiplicative decay floored at close (not additive)
# ---------------------------------------------------------------------------


class TestEdecayFormula(TestCase):
    """edecay uses multiplicative exp decay: result[i] = max(close[i], prev * exp(-1/n))."""

    def test_edecay_decays_after_spike(self):
        """After a spike, edecay must decay exponentially, not stay flat."""
        length = 5
        # Flat series then a spike: value should decay after the spike
        close = pd.Series([100.0] * 10 + [200.0] + [100.0] * 10)
        result = ta.edecay(close, length=length)
        self.assertIsNotNone(result)

        # Indices 11-13: decay path still > 100 floor (floor takes over at ~14)
        post_spike = result.iloc[11:14].values
        self.assertTrue(
            (post_spike > 100.0).all(),
            "edecay must stay above floor (close) right after spike",
        )
        self.assertTrue(
            (post_spike < 200.0).all(),
            "edecay must decay below spike peak",
        )

    def test_edecay_decay_rate_matches_formula(self):
        """Decay rate must match exp(-1/length) when close is below decay path."""
        import math

        length = 5
        factor = math.exp(-1.0 / length)
        # Spike then flat low floor — ensures decay path dominates
        close_vals = [10.0] * 5 + [1000.0] + [10.0] * 20
        close = pd.Series(close_vals)
        result = ta.edecay(close, length=length)

        # Two consecutive decaying bars: ratio must equal factor
        i = 6  # first bar after spike where decay path > close
        j = 7
        # Assert preconditions explicitly so a broken edecay can't silently pass
        self.assertGreater(
            result.iloc[i],
            10.0,
            f"Precondition failed: result[{i}]={result.iloc[i]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        self.assertGreater(
            result.iloc[j],
            10.0,
            f"Precondition failed: result[{j}]={result.iloc[j]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        ratio = result.iloc[j] / result.iloc[i]
        self.assertAlmostEqual(
            ratio,
            factor,
            places=6,
            msg=f"edecay ratio {ratio:.6f} must equal exp(-1/{length})={factor:.6f}",
        )

    def test_edecay_never_below_close(self):
        """edecay result must always be >= close (floored at close)."""
        close = pd.Series([50.0 + i % 20 for i in range(50)], dtype=float)
        result = ta.edecay(close, length=5)
        mask = result.notna()
        self.assertTrue(
            (result[mask].values >= close[mask].values - 1e-9).all(),
            "edecay must never fall below close (floor at close)",
        )


# ---------------------------------------------------------------------------
# Fix 10: psl open_ branch — verify_series(open_) returning None must propagate
# ---------------------------------------------------------------------------


class TestPslNoneGuard(TestCase):
    """psl(close, open_=too_short_series) must return None, not crash."""

    def test_psl_open_none_guard_returns_none(self):
        """When open_ is non-None but verify_series rejects it, psl must return None.

        verify_series returns None for non-Series inputs. Before the fix the
        function would proceed and crash; now it propagates None correctly.
        """
        close = pd.Series([float(i) for i in range(15)])
        # Pass a plain list — not a pd.Series — so verify_series returns None
        open_bad = [99.0] * 15
        result = ta.psl(close, open_=open_bad)
        self.assertIsNone(
            result,
            "psl must return None when open_ fails verify_series, not crash",
        )

    def test_psl_valid_open_returns_series(self):
        """psl with valid open_ must return a Series named PSL_{length}."""
        n = 20
        close = pd.Series([100.0 + i * 0.5 for i in range(n)])
        open_ = pd.Series([99.5 + i * 0.5 for i in range(n)])
        result = ta.psl(close, open_=open_, length=12)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.name.startswith("PSL_"))

    def test_psl_no_open_uses_diff(self):
        """psl without open_ must still return a valid Series."""
        close = pd.Series([100.0 + i for i in range(20)])
        result = ta.psl(close, length=12)
        self.assertIsNotNone(result)
        self.assertGreater(result.notna().sum(), 0)


# ---------------------------------------------------------------------------
# Fix 11: apply_fill honoured by indicators that previously ignored fillna kwarg
# ---------------------------------------------------------------------------


class TestApplyFillHonoured(TestCase):
    """fillna kwarg must be respected by indicators that previously silently ignored it."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def _assert_fillna_removes_nans(self, result, indicator_name):
        self.assertIsNotNone(result, f"{indicator_name} returned None unexpectedly")
        if isinstance(result, pd.DataFrame):
            nan_count = int(result.isna().sum().sum())
        else:
            nan_count = int(result.isna().sum())
        self.assertEqual(
            nan_count,
            0,
            f"{indicator_name}(fillna=0) must have 0 NaN values; got {nan_count}",
        )

    def test_hl2_fillna_removes_nans(self):
        result = ta.hl2(self.df["high"], self.df["low"], offset=5, fillna=0)
        self._assert_fillna_removes_nans(result, "hl2")

    def test_hlc3_fillna_removes_nans(self):
        result = ta.hlc3(
            self.df["high"],
            self.df["low"],
            self.df["close"],
            talib=False,
            offset=5,
            fillna=0,
        )
        self._assert_fillna_removes_nans(result, "hlc3")

    def test_avgprice_fillna_removes_nans(self):
        result = ta.avgprice(
            self.df["open"],
            self.df["high"],
            self.df["low"],
            self.df["close"],
            offset=5,
            fillna=0,
        )
        self._assert_fillna_removes_nans(result, "avgprice")

    def test_emv_fillna_removes_nans(self):
        result = ta.emv(
            self.df["high"],
            self.df["low"],
            self.df["volume"],
            offset=5,
            fillna=0,
        )
        self._assert_fillna_removes_nans(result, "emv")

    def test_edecay_fillna_removes_nans(self):
        result = ta.edecay(self.df["close"], offset=5, fillna=0)
        self._assert_fillna_removes_nans(result, "edecay")


# ---------------------------------------------------------------------------
# Fix 12: is_datetime_ordered returns bool (no return-in-finally SyntaxWarning)
# ---------------------------------------------------------------------------


class TestIsDatetimeOrdered(TestCase):
    """is_datetime_ordered must return a plain bool in all code paths."""

    def _fn(self):
        from pandas_ta_classic.utils import is_datetime_ordered

        return is_datetime_ordered

    def test_ordered_datetime_index_returns_true(self):
        fn = self._fn()
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series(range(5), index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_reverse_datetime_index_returns_false(self):
        fn = self._fn()
        idx = pd.date_range("2020-01-01", periods=5, freq="D")[::-1]
        s = pd.Series(range(5), index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_non_datetime_index_returns_false(self):
        fn = self._fn()
        s = pd.Series(range(5))  # integer index
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_single_element_returns_false(self):
        fn = self._fn()
        idx = pd.DatetimeIndex(["2020-01-01"])
        s = pd.Series([1.0], index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Fix 13: ad — invalid optional open_ returns None (not crash)
# Fix 14: ad — required-arg None guard added
# ---------------------------------------------------------------------------


class TestAdNoneGuard(TestCase):
    """ad() returns None gracefully on invalid inputs instead of crashing."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.close = cls.df["close"]
        cls.volume = cls.df["volume"]

    def test_required_arg_none_returns_none(self):
        """Passing None for required high → None, not AttributeError."""
        result = ta.ad(high=None, low=self.low, close=self.close, volume=self.volume)
        self.assertIsNone(result)

    def test_invalid_open_list_returns_none(self):
        """Passing a plain list as open_ triggers verify_series → None guard."""
        bad_open = list(self.close.values)
        result = ta.ad(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            open_=bad_open,
            talib=False,  # force else-branch where open_ is validated
        )
        self.assertIsNone(result)

    def test_valid_call_no_open_returns_series(self):
        """Normal call without open_ returns a Series named 'AD'."""
        result = ta.ad(high=self.high, low=self.low, close=self.close, volume=self.volume)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "AD")

    def test_valid_call_with_open_returns_series(self):
        """Normal call with valid open_ returns a Series named 'ADo'."""
        result = ta.ad(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            open_=self.df["open"],
        )
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "ADo")


# ---------------------------------------------------------------------------
# Fix 15: cmf — invalid optional open_ returns None (not crash)
# ---------------------------------------------------------------------------


class TestCmfNoneGuard(TestCase):
    """cmf() returns None gracefully on invalid optional open_."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.close = cls.df["close"]
        cls.volume = cls.df["volume"]

    def test_invalid_open_list_returns_none(self):
        """Passing a plain list as open_ triggers verify_series → None guard."""
        bad_open = list(self.close.values)
        result = ta.cmf(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            open_=bad_open,
        )
        self.assertIsNone(result)

    def test_valid_call_no_open_returns_series(self):
        """Normal call without open_ returns a Series."""
        result = ta.cmf(high=self.high, low=self.low, close=self.close, volume=self.volume)
        self.assertIsInstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Fix 16: psar — invalid optional close returns None (not crash)
# ---------------------------------------------------------------------------


class TestPsarCloseNoneGuard(TestCase):
    """psar() optional close None guard: invalid close → None, not AttributeError."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]

    def test_invalid_close_list_returns_none(self):
        """Passing a plain list as close triggers verify_series → None guard."""
        bad_close = list(self.high.values)
        result = ta.psar(high=self.high, low=self.low, close=bad_close)
        self.assertIsNone(result)

    def test_valid_call_without_close_returns_dataframe(self):
        """Normal call without close returns a DataFrame."""
        result = ta.psar(high=self.high, low=self.low)
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 17: talib=False propagated to sub-indicator calls
# dema, tema, t3, trima, cci, natr
# ---------------------------------------------------------------------------


class TestTalibFalsePropagation(TestCase):
    """talib=False is now forwarded to all sub-indicator calls.

    Each indicator is tested with talib=False to confirm:
    1. It returns a valid Series/DataFrame (no crash).
    2. The result is numerically consistent with direct pure-Python sub-calls.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.close = cls.df["close"]
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]

    def test_dema_talib_false_returns_series(self):
        """dema(talib=False) returns a Series."""
        result = ta.dema(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_dema_talib_false_matches_manual_calculation(self):
        """dema(talib=False) == 2*ema(talib=False) - ema(ema(talib=False)).

        EMA2 must be seeded from the first valid bar of EMA1 (leading NaN
        stripped) to match TA-Lib's lookback of 2*(length-1).
        """
        from pandas_ta_classic.overlap.ema import ema

        length = 10
        ema1 = ema(close=self.close, length=length, talib=False)
        ema2 = ema(close=ema1.loc[ema1.first_valid_index() :], length=length, talib=False)
        expected = 2 * ema1 - ema2
        result = ta.dema(self.close, length=length, talib=False)
        # drop leading NaNs then compare
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result[mask].values, expected[mask].values, rtol=1e-10)

    def test_tema_talib_false_returns_series(self):
        """tema(talib=False) returns a Series."""
        result = ta.tema(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_t3_talib_false_returns_series(self):
        """t3(talib=False) returns a Series."""
        result = ta.t3(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_trima_talib_false_returns_series(self):
        """trima(talib=False) returns a Series."""
        result = ta.trima(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_trima_talib_false_matches_manual_calculation(self):
        """trima(talib=False) == sma(sma(close, len1), len2) matching TA-Lib windows."""
        from pandas_ta_classic.overlap.sma import sma

        length = 10
        len1 = length // 2 + 1  # ceil((length+1)/2) — matches TA-Lib
        len2 = length // 2 + 1  # floor(length/2) + 1
        sma1 = sma(self.close, length=len1, talib=False)
        expected = sma(sma1, length=len2, talib=False)
        result = ta.trima(self.close, length=length, talib=False)
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result[mask].values, expected[mask].values, rtol=1e-10)

    def test_cci_talib_false_returns_series(self):
        """cci(talib=False) returns a Series."""
        result = ta.cci(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_natr_talib_false_returns_series(self):
        """natr(talib=False) returns a Series."""
        result = ta.natr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Fix 18: dm short-input guard
# ---------------------------------------------------------------------------


class TestDmShortInputGuard(TestCase):
    """dm() passed no min_length to verify_series, so its guard never fired.

    Every comparable indicator (adx, plus_dm, minus_dm) returns None when the
    input is shorter than `length`; dm returned a DataFrame computed from too
    few rows.
    """

    @classmethod
    def setUpClass(cls):
        df = get_sample_data()
        cls.high = df["high"]
        cls.low = df["low"]

    def test_short_input_returns_none(self):
        """Fewer rows than the default length of 14 → None."""
        result = ta.dm(self.high.iloc[:3], self.low.iloc[:3])
        self.assertIsNone(result)

    def test_short_input_matches_sibling_indicators(self):
        """dm agrees with plus_dm/minus_dm/adx on the same short input."""
        high, low = self.high.iloc[:3], self.low.iloc[:3]
        self.assertIsNone(ta.dm(high, low))
        self.assertIsNone(ta.plus_dm(high, low))
        self.assertIsNone(ta.minus_dm(high, low))

    def test_explicit_length_still_guarded(self):
        """An explicit length longer than the input is guarded too."""
        self.assertIsNone(ta.dm(self.high.iloc[:10], self.low.iloc[:10], length=20))

    def test_sufficient_input_still_returns_dataframe(self):
        """Normal-length input is unaffected."""
        result = ta.dm(self.high, self.low)
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 19: cdl_pattern None-check on pta_patterns results
# ---------------------------------------------------------------------------


class TestCdlPatternShortInput(TestCase):
    """cdl_pattern() dereferenced `.name` on a sub-pattern result without checking it.

    `cdl_doji` has a default length of 10, so on a short frame it returns None
    and the aggregation raised `AttributeError: 'NoneType' object has no
    attribute 'name'`. The native-pattern branch already had this guard.
    """

    @classmethod
    def setUpClass(cls):
        df = get_sample_data()
        cls.open_ = df["open"].iloc[:3]
        cls.high = df["high"].iloc[:3]
        cls.low = df["low"].iloc[:3]
        cls.close = df["close"].iloc[:3]

    def test_short_input_does_not_raise(self):
        """Short input returns a result instead of raising."""
        result = ta.cdl_pattern(self.open_, self.high, self.low, self.close)
        self.assertIsInstance(result, pd.DataFrame)

    def test_uncomputable_subpattern_is_skipped(self):
        """doji needs 10 rows, so its column is absent rather than fatal."""
        result = ta.cdl_pattern(self.open_, self.high, self.low, self.close)
        self.assertNotIn("CDL_DOJI_10", result.columns)
        self.assertIn("CDL_INSIDE", result.columns)

    def test_named_uncomputable_pattern_returns_none(self):
        """Asking only for a pattern that cannot be computed yields None."""
        result = ta.cdl_pattern(self.open_, self.high, self.low, self.close, name="doji")
        self.assertIsNone(result)

    def test_accessor_path_does_not_raise(self):
        """The same call through df.ta.cdl_pattern() is fixed as well."""
        df = pd.DataFrame(
            {
                "open": self.open_,
                "high": self.high,
                "low": self.low,
                "close": self.close,
            }
        )
        result = df.ta.cdl_pattern()
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 20: ema SMA-seed bounds check
# ---------------------------------------------------------------------------


class TestEmaSeedBounds(TestCase):
    """ema() seeded at `fv_pos + length - 1` without checking that it exists.

    verify_series' min_length models a single window, but chained indicators
    compound their lookback: trix applies ema three times, so each inner call's
    NaN prefix pushes the seed position further out. Once it passed the end of
    the series, `.iloc[]` raised "IndexError: iloc cannot enlarge its target
    object" on clean data with default arguments.
    """

    def test_trix_on_thirty_clean_rows(self):
        """The minimal user-facing reproducer: 30 rows, defaults, no NaN."""
        result = ta.trix(pd.Series(range(1, 31), dtype=float))
        self.assertIsInstance(result, pd.DataFrame)

    def test_chained_ema_over_nan_prefix(self):
        """A NaN prefix from an inner call no longer pushes the seed off the end."""
        close = pd.Series(np.arange(1.0, 201.0))
        result = ta.ema(ta.sma(close, length=190), length=20)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.isna().all(), "undefined EMA must be all-NaN")

    def test_affected_indicators_no_longer_raise(self):
        """Each indicator, at a row count that previously raised."""
        cases = {
            "efi": 13, "inertia": 20, "kc": 20, "pgo": 14, "ppo": 26,
            "pvo": 26, "qqe": 27, "rvi": 14, "smi": 20, "thermo": 20,
            "trix": 30, "trixh": 18, "tsi": 25, "zlma": 10,
        }
        rng = np.random.default_rng(0)
        for name, rows in cases.items():
            with self.subTest(indicator=name, rows=rows):
                base = 100 + np.cumsum(rng.normal(0, 1, rows))
                index = pd.date_range("2020-01-01", periods=rows, freq="D")
                func = getattr(ta, name)
                kwargs = {}
                for param in inspect.signature(func).parameters:
                    if param == "close":
                        kwargs[param] = pd.Series(base, index=index)
                    elif param == "high":
                        kwargs[param] = pd.Series(base + 1.0, index=index)
                    elif param == "low":
                        kwargs[param] = pd.Series(base - 1.0, index=index)
                    elif param == "open_":
                        kwargs[param] = pd.Series(base - 0.5, index=index)
                    elif param == "volume":
                        kwargs[param] = pd.Series(
                            rng.integers(1_000, 5_000, rows).astype(float), index=index
                        )
                func(**kwargs)  # must not raise

    def test_strategy_on_short_frame(self):
        """df.ta.strategy("all") on 20 clean rows previously raised IndexError."""
        rows = 20
        rng = np.random.default_rng(0)
        base = 100 + np.cumsum(rng.normal(0, 1, rows))
        df = pd.DataFrame(
            {
                "open": base - 0.5,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base,
                "volume": rng.integers(1_000, 5_000, rows).astype(float),
            },
            index=pd.date_range("2020-01-01", periods=rows, freq="D"),
        )
        df.ta.strategy("all", cores=0)  # must not raise

    def test_ample_input_seed_unchanged(self):
        """Normal-length input keeps the TA-Lib lookback: length-1 leading NaN."""
        close = pd.Series(np.arange(1.0, 201.0))
        result = ta.ema(close, length=20)
        self.assertEqual(result.isna().sum(), 19)
        self.assertTrue(np.isfinite(result.iloc[19:]).all())

    def test_bbands_with_zlma_mamode(self):
        """bbands(mamode="zlma") on 5 rows previously raised through ema."""
        result = ta.bbands(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), mamode="zlma")
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 21: _hilbert NaN handling
# ---------------------------------------------------------------------------


class TestHilbertNanInput(TestCase):
    """The Hilbert Transform loop crashed in int(nan) on any NaN input.

    `dc_period_int = max(int(sp + 0.5), 1)` raised "ValueError: cannot convert
    float NaN to integer" once a NaN reached the smoothed period, which a single
    NaN anywhere in the input guarantees. This contradicted the documented
    contract that an indicator may propagate NaN but must not crash.
    """

    HT_NAMES = (
        "ht_trendline",
        "ht_dcperiod",
        "ht_dcphase",
        "ht_sine",
        "ht_phasor",
        "ht_trendmode",
    )

    @classmethod
    def setUpClass(cls):
        cls.clean = pd.Series(100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 200)))
        cls.with_nan = cls.clean.copy()
        cls.with_nan.iloc[50] = np.nan

    def test_single_nan_does_not_raise(self):
        """One NaN mid-series returns a result for every ht_* indicator."""
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                result = getattr(ta, name)(self.with_nan)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, (pd.Series, pd.DataFrame))

    def test_nan_propagates_after_the_gap(self):
        """The recursion is poisoned from the NaN onward, so output goes NaN."""
        result = ta.ht_trendline(self.with_nan)
        self.assertTrue(result.iloc[54:].isna().all())

    def test_values_before_the_gap_are_unaffected(self):
        """Bars before the NaN keep the values computed from clean input."""
        clean_result = ta.ht_trendline(self.clean)
        nan_result = ta.ht_trendline(self.with_nan)
        mask = clean_result.iloc[:47].notna()
        np.testing.assert_allclose(
            nan_result.iloc[:47][mask].values,
            clean_result.iloc[:47][mask].values,
            rtol=1e-12,
        )

    def test_inf_does_not_raise(self):
        """±Inf is the other half of the documented edge-case contract."""
        with_inf = self.clean.copy()
        with_inf.iloc[50] = np.inf
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                self.assertIsNotNone(getattr(ta, name)(with_inf))

    def test_clean_input_still_produces_values(self):
        """No NaN anywhere: the guard must not fire."""
        result = ta.ht_trendline(self.clean)
        self.assertGreater(int(np.isfinite(result).sum()), 100)


# ---------------------------------------------------------------------------
# Fix 22: candle_color NaN handling
# ---------------------------------------------------------------------------


class TestCandleColorNanInput(TestCase):
    """candle_color() did `close.copy().astype(int)`, which raises on NaN.

    pandas cannot cast NaN to int, so `IntCastingNaNError` propagated out of
    cdl_inside and cdl_pattern for any input carrying a NaN -- which is what
    every chained indicator produces.
    """

    def test_clean_input_keeps_int_dtype(self):
        """Fully defined input keeps the historical integer dtype."""
        open_ = pd.Series([1.0, 2.0, 3.0])
        close = pd.Series([2.0, 1.0, 3.0])
        result = ta.candle_color(open_, close)
        self.assertEqual(result.dtype, np.dtype("int64"))
        self.assertEqual(result.tolist(), [1, -1, 1])

    def test_nan_yields_nan_not_a_colour(self):
        """An undefined candle is NaN, not silently classified as bearish."""
        open_ = pd.Series([1.0, 2.0, 3.0])
        close = pd.Series([2.0, np.nan, 3.0])
        result = ta.candle_color(open_, close)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertEqual(result.iloc[0], 1.0)
        self.assertEqual(result.iloc[2], 1.0)

    def test_cdl_inside_with_nan_does_not_raise(self):
        """cdl_inside previously raised IntCastingNaNError."""
        close = pd.Series(np.arange(1.0, 201.0))
        close.iloc[50] = np.nan
        result = ta.cdl_inside(close, close + 1, close - 1, close)
        self.assertIsInstance(result, pd.Series)

    def test_cdl_pattern_with_nan_does_not_raise(self):
        """The aggregation path is fixed as well."""
        close = pd.Series(np.arange(1.0, 201.0))
        close.iloc[50] = np.nan
        result = ta.cdl_pattern(close - 0.5, close + 1, close - 1, close)
        self.assertIsInstance(result, pd.DataFrame)

    def test_asbool_path_unaffected(self):
        """asbool=True never calls candle_color."""
        close = pd.Series(np.arange(1.0, 51.0))
        result = ta.cdl_inside(close - 0.5, close + 1, close - 1, close, asbool=True)
        self.assertEqual(result.dtype, np.dtype("bool"))


# ---------------------------------------------------------------------------
# Fix 23: short-window hardening for rma, linreg and _sliding_weighted_ma
# ---------------------------------------------------------------------------


class TestShortWindowHardening(TestCase):
    """Three sites that crashed when handed a window longer than the series.

    None of them is reachable through the public API: verify_series' min_length
    guard blocks every route found while probing (oversized single and multi
    window arguments, every mamode, the accessor and strategy paths, NaN-prefixed
    input, and chained calls). They are hardened regardless, because the ema
    seeding bug (fix 20) was equally unreachable until someone worked out which
    composition of indicators exposed it -- min_length models a single window,
    while real lookback is compositional.

    Reaching the guarded branches therefore requires bypassing verify_series,
    which is what these tests do.
    """

    @staticmethod
    def _passthrough(series, min_length=None):
        """verify_series without the min_length short-circuit."""
        return series if isinstance(series, pd.Series) else None

    def _without_guard(self, dotted_name):
        """Swap verify_series in a module for the pass-through, restoring after.

        The module has to be resolved by name: ``pandas_ta_classic.overlap``
        re-exports each indicator function under the same name as its submodule,
        so plain attribute access returns the function, not the module.
        """
        module = importlib.import_module(dotted_name)
        original = module.verify_series
        module.verify_series = self._passthrough
        self.addCleanup(setattr, module, "verify_series", original)
        return module

    def test_sliding_weighted_ma_window_longer_than_series(self):
        """The helper is directly callable and must not build an empty view."""
        from pandas_ta_classic.utils._core import _sliding_weighted_ma

        close = pd.Series([1.0, 2.0, 3.0])
        result = _sliding_weighted_ma(close, 10, np.ones(10))
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.isna().all())

    def test_sliding_weighted_ma_exact_fit(self):
        """length == len(series) still produces the single valid window."""
        from pandas_ta_classic.utils._core import _sliding_weighted_ma

        close = pd.Series([1.0, 2.0, 3.0])
        result = _sliding_weighted_ma(close, 3, np.ones(3))
        self.assertEqual(result.iloc[-1], 6.0)
        self.assertTrue(result.iloc[:2].isna().all())

    def test_rma_without_guard(self):
        """rma seeded at iloc[length - 1] without checking it exists."""
        module = self._without_guard("pandas_ta_classic.overlap.rma")
        result = module.rma(pd.Series([1.0, 2.0, 3.0]), length=10)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.isna().all())

    def test_linreg_without_guard(self):
        """linreg built a sliding window wider than the input array."""
        module = self._without_guard("pandas_ta_classic.overlap.linreg")
        result = module.linreg(pd.Series([1.0, 2.0, 3.0]), length=10, talib=False)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.isna().all())

    def test_alma_without_guard(self):
        """alma reaches _sliding_weighted_ma, so it is covered too."""
        module = self._without_guard("pandas_ta_classic.overlap.alma")
        result = module.alma(pd.Series([1.0, 2.0, 3.0]), length=10)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.isna().all())

    def test_valid_input_unaffected(self):
        """Ample input keeps its usual lookback in all three."""
        close = pd.Series(np.arange(1.0, 101.0))
        self.assertEqual(ta.rma(close, length=10).isna().sum(), 9)
        self.assertEqual(ta.linreg(close, length=10, talib=False).isna().sum(), 9)
        self.assertEqual(ta.alma(close, length=10).isna().sum(), 9)
