from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.assertions import assert_all_nan
from tests.config import get_sample_data


class TestMomentum(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_datetime_ordered(self):
        # Test if datetime64 index and ordered
        result = self.data.ta.datetime_ordered
        self.assertTrue(result)

        # Test if not ordered
        original = self.data.copy()
        reversal = original.ta.reverse
        result = reversal.ta.datetime_ordered
        self.assertFalse(result)

        # Test a non-datetime64 index
        original = self.data.copy()
        original = original.reset_index()
        result = original.ta.datetime_ordered
        self.assertFalse(result)

    def test_reverse(self):
        original = self.data.copy()
        result = original.ta.reverse

        # Check if first and last time are reversed
        self.assertEqual(result.index[-1], original.index[0])
        self.assertEqual(result.index[0], original.index[-1])

    def test_cmo(self):
        # The default (talib=False) is Chande's rolling sum, which TA-Lib does not implement,
        # so it is checked against that definition; the Wilder path (talib=True) is in
        # tests/oracle/test_oracle_registry.py.
        mom = self.close.diff(1)
        pos = mom.clip(lower=0).rolling(14).sum()
        neg = mom.clip(upper=0).abs().rolling(14).sum()
        chande = 100 * (pos - neg) / (pos + neg)
        np.testing.assert_allclose(pandas_ta.cmo(self.close, talib=False).to_numpy(), chande.to_numpy(), rtol=1e-12, atol=1e-9)

    def test_inertia(self):
        result = pandas_ta.inertia(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAr_20_14")

        result = pandas_ta.inertia(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAt_20_14")

        # refined/thirds used to be bool()-coerced and were both silently
        # optional: inertia(close, refined=True) returned None rather than
        # saying it needs high/low.
        for kwarg in ("refined", "thirds"):
            with self.subTest(kwarg=kwarg):
                with self.assertRaisesRegex(ValueError, rf"inertia\(\) {kwarg} must be True or False"):
                    pandas_ta.inertia(self.close, self.high, self.low, **{kwarg: "yes"})
                with self.assertRaisesRegex(ValueError, rf"inertia\(\) {kwarg}=True needs both high and low"):
                    pandas_ta.inertia(self.close, **{kwarg: True})
        with self.assertRaisesRegex(ValueError, "alternative modes"):
            pandas_ta.inertia(self.close, self.high, self.low, refined=True, thirds=True)

    def test_macdext(self):
        # EMA-based should closely match regular MACD
        result = pandas_ta.macdext(self.close, talib=False)
        # SMA-based should give different values
        result_sma = pandas_ta.macdext(self.close, fastmatype=0, slowmatype=0, signalmatype=0, talib=False)
        self.assertIsInstance(result_sma, DataFrame)
        # Non-default MA types change the values, so they are in the name (#343)
        self.assertEqual(result_sma.columns[0], "MACDEXT_12_26_9_fastmatype0_slowmatype0_signalmatype0")
        self.assertFalse(result["MACDEXT_12_26_9"].equals(result_sma.iloc[:, 0]))
        if pandas_ta.Imports["talib"]:
            result_tal = pandas_ta.macdext(self.close)
            self.assertIsInstance(result_tal, DataFrame)
            self.assertEqual(result_tal.name, "MACDEXT_12_26_9")
        pandas_ta.macdext(self.close, fillna=0)
        pandas_ta.macdext(self.close, fill_method="ffill")

    def test_macdext_unsupported_matype_raises(self):
        # KAMA/MAMA have no native path; an EMA used to be computed instead
        with self.assertRaisesRegex(ValueError, r"signalmatype=6 .*needs TA-Lib"):
            pandas_ta.macdext(self.close, signalmatype=6, talib=False)
        with self.assertRaisesRegex(ValueError, r"fastmatype=7 .*needs TA-Lib"):
            pandas_ta.macdext(self.close, fastmatype=7, talib=False)

    def test_squeeze(self):
        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5")

        result = pandas_ta.squeeze(self.high, self.low, self.close, lazybear=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5_LB")

        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False, lazybear=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5_LB")

    def test_squeeze_detailed(self):
        base = pandas_ta.squeeze(self.high, self.low, self.close)
        result = pandas_ta.squeeze(self.high, self.low, self.close, detailed=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5")

        detail_columns = ["SQZ_INC", "SQZ_DEC", "SQZ_PINC", "SQZ_PDEC", "SQZ_NDEC", "SQZ_NINC"]
        self.assertEqual(list(result.columns), list(base.columns) + detail_columns)
        # detailed=True only appends; the base columns keep their values
        for col in base.columns:
            self.assertTrue(result[col].equals(base[col]), col)

        squeeze_series = result["SQZ_20_2.0_20_1.5"]
        for col in detail_columns:
            # Every detail column masks the bars it does not describe, so it is
            # neither all NaN nor fully populated, and never carries a 0.
            self.assertTrue(result[col].isna().any(), col)
            self.assertTrue(result[col].notna().any(), col)
            self.assertFalse((result[col] == 0).any(), col)
            # Where a detail column has a value it is the squeeze value itself.
            defined = result[col].notna()
            self.assertTrue(result.loc[defined, col].equals(squeeze_series[defined]), col)

        # SQZ_INC / SQZ_DEC partition the defined bars into rising and falling.
        self.assertFalse((result["SQZ_INC"].notna() & result["SQZ_DEC"].notna()).any())
        # PINC/PDEC only describe non-negative bars, NINC/NDEC only negative ones.
        for col in ("SQZ_PINC", "SQZ_PDEC"):
            self.assertTrue((result.loc[result[col].notna(), col] >= 0).all(), col)
        for col in ("SQZ_NINC", "SQZ_NDEC"):
            self.assertTrue((result.loc[result[col].notna(), col] < 0).all(), col)

        # lazybear and the accessor reach the same detailed branch
        lb = pandas_ta.squeeze(self.high, self.low, self.close, detailed=True, lazybear=True)
        self.assertEqual(list(lb.columns), ["SQZ_20_2.0_20_1.5_LB", "SQZ_ON", "SQZ_OFF", "SQZ_NO"] + detail_columns)

        via_accessor = self.data.ta.squeeze(detailed=True)
        self.assertEqual(list(via_accessor.columns), list(result.columns))

    def test_squeeze_pro(self):
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_2_1.5_1")

        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPRO_20_2.0_20_3_2.0_1")

        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_3_2.0_1")

    def test_stc_reads_nan_until_its_windows_hold_data(self):
        # The recursion seeds 0.0 and held it through the warm-up, so STC read
        # 0.0 from bar 0 while its own MACD column was still NaN. MACD starts at
        # slow - 1 = 25, its 10-bar range at 34 (STCstoch), and STC's own window
        # of real STCstoch values at 43; later values are the recursion's, unchanged.
        result = pandas_ta.stc(self.close)
        first = {column: int(result[column].notna().to_numpy().argmax()) for column in result.columns}
        self.assertEqual(first, {"STC_10_12_26_0.5": 43, "STCmacd_10_12_26_0.5": 25, "STCstoch_10_12_26_0.5": 34})
        self.assertFalse(result.iloc[43:].isna().any().any())

    def test_stc_short_series_through_kwargs_raises(self):
        # A series shorter than close does not share its index: it used to read
        # all-NaN here, a longer one failed inside pandas, and a shifted one was
        # aligned by label in one column and by position in the others (#341).
        ma1, ma2 = pandas_ta.ema(self.close, length=12), pandas_ta.ema(self.close, length=26)
        for kwargs, key in (({"osc": (ma1 - ma2).iloc[:5]}, "osc"), ({"ma1": ma1.iloc[:5], "ma2": ma2.iloc[:5]}, "ma1")):
            with self.subTest(kwargs=sorted(kwargs)), self.assertRaisesRegex(ValueError, rf"^stc\(\) {key} must have the same index as close: {key} has 5 rows"):
                pandas_ta.stc(self.close, **kwargs)

    def test_stc_external_series(self):
        traditional = pandas_ta.stc(self.close)
        macd_col = "STCmacd_10_12_26_0.5"

        # ma1 + ma2 replace the internal EMA pair: feeding the same EMAs back in
        # reproduces the traditional result exactly.
        ma1, ma2 = pandas_ta.ema(self.close, length=12), pandas_ta.ema(self.close, length=26)
        with_mas = pandas_ta.stc(self.close, ma1=ma1, ma2=ma2)
        self.assertIsInstance(with_mas, DataFrame)
        self.assertTrue(with_mas.equals(traditional))

        # osc replaces both MAs with a ready-made oscillator.
        with_osc = pandas_ta.stc(self.close, osc=ma1 - ma2)
        self.assertTrue(with_osc.equals(traditional))
        # osc wins over ma1/ma2 when all three are given.
        self.assertTrue(pandas_ta.stc(self.close, ma1=ma1, ma2=ma2, osc=ma1 - ma2).equals(traditional))

        # A distinct oscillator must change the macd column.
        other = pandas_ta.stc(self.close, osc=pandas_ta.ema(self.close, length=5) - ma2)
        self.assertFalse(other[macd_col].equals(traditional[macd_col]))

        # ma1 and ma2 are only meaningful as a pair.
        for kwargs in ({"ma1": ma1}, {"ma2": ma2}):
            with self.assertRaises(ValueError):
                pandas_ta.stc(self.close, **kwargs)

        # slow < fast was swapped silently; it raises now (#374).
        with self.assertRaisesRegex(ValueError, r"^stc\(\) needs fast <= slow, got fast=26, slow=12$"):
            pandas_ta.stc(self.close, fast=26, slow=12)

    def test_willr_guards_a_zero_range(self):
        """willr was the only rolling-range divider without non_zero_range.

        A window where every bar shares its high and low divided 0/0, and the
        NaN then spread over the next length-1 windows: a flat block of b bars
        cost max(0, b - (length - 1)) extra NaN and a fully flat series was all
        NaN. The six siblings (stoch, stochf, stochrsi, kdj, fisher, stc) read
        0.0 on a degenerate window too, each through its own zero-denominator
        guard rather than non_zero_range.
        """
        length = 14
        flat = Series([100.0] * 120)
        result = pandas_ta.willr(flat, flat, flat, length=length)
        # Only the warm-up is NaN now, and the window reads 0.0 -- TA-Lib's
        # marker for a degenerate window, which this package follows
        # throughout. -100 cannot double as the marker: it is a real %R value
        # for a bar sitting at the low of a real range. See
        # tests/contracts/test_degenerate_input.py.
        self.assertEqual(int(result.isna().sum()), length - 1)
        self.assertTrue((result.dropna() == 0.0).all())

        # A flat block no longer bleeds NaN into the windows after it.
        high, low, close = self.high.copy(), self.low.copy(), self.close.copy()
        for block in (5, 14, 20, 30):
            with self.subTest(block=block):
                h, low_, c = high.copy(), low.copy(), close.copy()
                h.iloc[80 : 80 + block] = low_.iloc[80 : 80 + block] = c.iloc[80 : 80 + block] = 100.0
                out = pandas_ta.willr(h, low_, c, length=length)
                self.assertEqual(int(out.loc[out.first_valid_index() :].isna().sum()), 0)

    def test_lrsi_has_no_value_on_its_seed_bar(self):
        # Bar 0 seeds every Laguerre filter with the close itself, so it read 0/0 = 0.0
        # on any data; the warm-up contract skips it, as 0 is also LRSI's lower bound.
        result = pandas_ta.lrsi(self.close)
        self.assertTrue(result.isna().iloc[0])
        self.assertFalse(result.iloc[1:].isna().any())


# ---------------------------------------------------------------------------
# QQE column count (was 3, now >= 6 including QQEb_l, QQEb_s, QQEd)
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
        return pandas_ta.qqe(self.close)

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
# rvgi returns 3 columns including the histogram column (RVGIh_*)
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
        return pandas_ta.rvgi(self.df["open"], self.df["high"], self.df["low"], self.df["close"])

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
# psl open_ branch — verify_series(open_) returning None must propagate
# ---------------------------------------------------------------------------


class TestPslNoneGuard(TestCase):
    """psl(close, open_=too_short_series) must return None, not crash."""

    def test_psl_open_non_series_raises(self):
        """A non-Series open_ is rejected by verify_series, not used or crashed on.

        Before the fix psl proceeded past a rejected open_ and crashed. Since
        issue #145 verify_series raises TypeError for non-Series input.
        """
        close = pd.Series([float(i) for i in range(15)])
        open_bad = [99.0] * 15
        with self.assertRaisesRegex(TypeError, r"psl\(\) expected a pandas Series but got list"):
            pandas_ta.psl(close, open_=open_bad)

    def test_psl_valid_open_returns_series(self):
        """psl with valid open_ must return a Series named PSL_{length}."""
        n = 20
        close = pd.Series([100.0 + i * 0.5 for i in range(n)])
        open_ = pd.Series([99.5 + i * 0.5 for i in range(n)])
        result = pandas_ta.psl(close, open_=open_, length=12)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(result.name.startswith("PSL_"))

    def test_psl_no_open_uses_diff(self):
        """psl without open_ must still return a valid Series."""
        close = pd.Series([100.0 + i for i in range(20)])
        result = pandas_ta.psl(close, length=12)
        self.assertIsNotNone(result)
        self.assertGreater(result.notna().sum(), 0)


# ---------------------------------------------------------------------------
# dm short-input guard
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
        result = pandas_ta.dm(self.high.iloc[:3], self.low.iloc[:3])
        assert_all_nan(self, result)

    def test_short_input_matches_sibling_indicators(self):
        """dm agrees with plus_dm/minus_dm/adx on the same short input."""
        high, low = self.high.iloc[:3], self.low.iloc[:3]
        assert_all_nan(self, pandas_ta.dm(high, low))
        assert_all_nan(self, pandas_ta.plus_dm(high, low))
        assert_all_nan(self, pandas_ta.minus_dm(high, low))

    def test_explicit_length_still_guarded(self):
        """An explicit length longer than the input is guarded too."""
        assert_all_nan(self, pandas_ta.dm(self.high.iloc[:10], self.low.iloc[:10], length=20))

    def test_sufficient_input_still_returns_dataframe(self):
        """Normal-length input is unaffected."""
        result = pandas_ta.dm(self.high, self.low)
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# macd signal options reached only the histogram signals
# ---------------------------------------------------------------------------


class TestMacdSignalOptionsApplyToBothLines(TestCase):
    """The first signals() call popped xa/xb/xserie/cross_series from kwargs.

    The second call, for the MACD line, then read the defaults, so
    macd(signal_indicators=True, xa=1) produced MACD_12_26_9_A_0.
    """

    @classmethod
    def setUpClass(cls):
        cls.close = get_sample_data().close.iloc[:400]

    def _signal_columns(self, **kwargs):
        return list(pandas_ta.macd(self.close, signal_indicators=True, **kwargs).columns[3:])

    def test_defaults_unchanged(self):
        self.assertEqual(self._signal_columns(), ["MACDh_12_26_9_XA_0", "MACDh_12_26_9_XB_0", "MACD_12_26_9_A_0"])

    def test_thresholds_reach_the_macd_line(self):
        columns = self._signal_columns(xa=1, xb=-1)
        self.assertIn("MACD_12_26_9_A_1", columns)
        self.assertIn("MACD_12_26_9_B_-1", columns)
        self.assertNotIn("MACD_12_26_9_A_0", columns)

    def test_series_reaches_the_macd_line(self):
        xserie = self.close.rolling(5).mean() - self.close.rolling(20).mean()
        self.assertIn("MACD_12_26_9_XA_close", self._signal_columns(xserie=xserie))
        self.assertIn("MACD_12_26_9_A_close", self._signal_columns(xserie=xserie, cross_series=False))

    def test_flags_are_validated(self):
        with self.assertRaisesRegex(ValueError, r"macd\(\) cross_values must be True or False"):
            pandas_ta.macd(self.close, signal_indicators=True, cross_values="no")
