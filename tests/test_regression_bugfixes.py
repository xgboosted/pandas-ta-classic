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

Run:
    python -m unittest tests/test_regression_bugfixes.py
"""

import math
from unittest import TestCase

import numpy as np
import pandas as pd

import pandas_ta_classic as ta
from tests.config import get_sample_data
from tests.context import pandas_ta_classic  # noqa: F401


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
            "stdev() default must NOT equal sample std (ddof=1); "
            "ddof=0 (population) is the correct post-fix default",
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
            f"linreg(angle=True) max|value|={max_abs:.4f} must exceed "
            "pi/2 ({:.4f}), confirming values are in degrees".format(math.pi / 2),
        )

    def test_explicit_degrees_false_returns_radians(self):
        """linreg(angle=True, degrees=False) must return radian values in (-pi/2, pi/2)."""
        length = 14
        result = ta.linreg(
            self.close, length=length, angle=True, degrees=False, talib=False
        )
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
        deg = ta.linreg(
            self.close, length=length, angle=True, degrees=True, talib=False
        )
        rad = ta.linreg(
            self.close, length=length, angle=True, degrees=False, talib=False
        )
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
        h, l, v = self.high, self.low, self.volume
        hl_range = (h - l).replace(0, float("nan"))
        midpoint = 0.5 * (h + l)
        distance = midpoint - midpoint.shift(1)
        box_ratio = (v / 10_000) / hl_range
        expected = distance / box_ratio

        result = ta.emv(h, l, v)
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
        h, l, v = self.high, self.low, self.volume
        default = ta.emv(h, l, v)
        custom = ta.emv(h, l, v, divisor=1)

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
        return ta.rvgi(
            self.df["open"], self.df["high"], self.df["low"], self.df["close"]
        )

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
        mask = (
            result[rvgi_col].notna()
            & result[sig_col].notna()
            & result[hist_col].notna()
        )
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
        l = pd.Series([8.0, 9.0, 10.0])
        result = ta.hl2(h, l)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "HL2")
        np.testing.assert_allclose(result.values, [9.0, 10.0, 11.0], rtol=1e-9)

    def test_hlc3_valid_input_returns_series(self):
        """hlc3 with valid inputs must return a Series named 'HLC3'."""
        h = pd.Series([12.0, 13.0])
        l = pd.Series([8.0, 9.0])
        c = pd.Series([10.0, 11.0])
        result = ta.hlc3(h, l, c, talib=False)
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
        import math

        length = 5
        factor = math.exp(-1.0 / length)
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
            f"Precondition failed: result[{i}]={result.iloc[i]:.4f} must be > 10.0 "
            "(decay path not yet floored at close)",
        )
        self.assertGreater(
            result.iloc[j],
            10.0,
            f"Precondition failed: result[{j}]={result.iloc[j]:.4f} must be > 10.0 "
            "(decay path not yet floored at close)",
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
