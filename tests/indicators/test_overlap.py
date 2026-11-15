import inspect
import math
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestOverlap(TestCase):
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

    def test_ichimoku(self):
        # Default append_span=False: visible period only, no future-dated rows.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pandas_ta.ichimoku(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ICHIMOKU_9_26_52")
        self.assertEqual(result.category, "overlap")
        self.assertEqual(
            list(result.columns),
            ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
        )
        self.assertEqual(len(result), len(self.close))

    def test_ichimoku_append_span(self):
        kijun = 26
        result = pandas_ta.ichimoku(self.high, self.low, self.close, append_span=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ICHIMOKU_9_26_52")
        self.assertEqual(
            list(result.columns),
            ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
        )
        # Length = visible rows (len close) + future-dated span rows (kijun).
        self.assertEqual(len(result), len(self.close) + kijun)
        # Span rows: only ISA/ISB populated; ITS/IKS/ICS are NaN.
        span_rows = result.iloc[-kijun:]
        self.assertTrue(span_rows[["ITS_9", "IKS_26", "ICS_26"]].isna().all().all())
        self.assertTrue(span_rows["ISA_9"].notna().any())

    def test_ichimoku_as_dataframe_removed(self):
        # The (visible, span) tuple and its selector were removed in 0.9.0;
        # **kwargs would otherwise swallow the old keyword silently.
        for value in (True, False):
            with self.assertRaisesRegex(TypeError, "no longer accepts 'as_dataframe'"):
                pandas_ta.ichimoku(self.high, self.low, self.close, as_dataframe=value)

    def test_ma(self):
        result = pandas_ta.ma()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        result = pandas_ta.ma("ema", self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        result = pandas_ta.ma("fwma", self.close, length=15)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "FWMA_15")

    def _mavp_periods(self):
        """A per-bar window schedule; mavp has no default for this."""
        cycle = [5, 10, 20]
        values = [cycle[i % len(cycle)] for i in range(len(self.close))]
        return Series(values, index=self.close.index, dtype=float)

    def test_mavp_requires_periods(self):
        """`periods` has no causal default, so it is a required argument."""
        with self.assertRaises(TypeError):
            pandas_ta.mavp(self.close)

    def test_mavp_accessor_without_periods_raises(self):
        """The accessor fills missing required args with None; that must not pass silently."""
        with self.assertRaises(ValueError):
            self.data.ta.mavp()

    def test_mavp_unsupported_mamode_raises(self):
        # the native path computes only an SMA; it used to do so for any mamode
        with self.assertRaisesRegex(ValueError, r"mavp\(\) mamode=1 needs TA-Lib"):
            pandas_ta.mavp(self.close, periods=self._mavp_periods(), mamode=1, talib=False)

    def test_ssf(self):
        result = pandas_ta.ssf(self.close, poles=2)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_2")
        result = pandas_ta.ssf(self.close, poles=3)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_3")

    def test_weighted_ma_asc(self):
        """asc=False was forced back to True, so it returned the ascending result."""
        from pandas_ta_classic.utils import fibonacci

        close = self.close.iloc[:200]
        arr = close.to_numpy(dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(arr, 10)

        def expected(w):
            out = np.full(arr.size, np.nan)
            out[9:] = windows @ w
            return out

        desc = np.arange(10, 0, -1) / 55.0
        np.testing.assert_allclose(pandas_ta.wma(close, 10, asc=False).to_numpy(), expected(desc), rtol=1e-12)
        np.testing.assert_allclose(pandas_ta.wma(close, 10, asc=False, talib=True).to_numpy(), expected(desc), rtol=1e-12)
        fibs = fibonacci(n=10, weighted=True)[::-1]
        np.testing.assert_allclose(pandas_ta.fwma(close, 10, asc=False).to_numpy(), expected(fibs), rtol=1e-12)
        for name in ("wma", "fwma"):
            with self.subTest(name=name):
                fn = getattr(pandas_ta, name)
                self.assertFalse(fn(close, 10, asc=False).equals(fn(close, 10)))
        for name in ("pwma", "swma"):  # symmetric weights: asc was removed; strategy() hands it only to wma and fwma
            with self.subTest(name=name):
                fn = getattr(pandas_ta, name)
                self.assertNotIn("asc", inspect.signature(fn).parameters)
                with self.assertRaisesRegex(TypeError, rf"^{name}\(\) got an unexpected keyword argument 'asc'$"):
                    fn(close, 10, asc=False)


# ---------------------------------------------------------------------------
# linreg degrees default = True
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
        result = pandas_ta.linreg(self.close, length=length, angle=True, talib=False)
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
        result = pandas_ta.linreg(self.close, length=length, angle=True, degrees=False, talib=False)
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
        deg = pandas_ta.linreg(self.close, length=length, angle=True, degrees=True, talib=False)
        rad = pandas_ta.linreg(self.close, length=length, angle=True, degrees=False, talib=False)
        mask = deg.notna() & rad.notna()
        diff = (deg[mask] - rad[mask]).abs()
        self.assertTrue(
            diff.max() > 1e-6,
            "degrees=True and degrees=False must produce different angle values",
        )


# ---------------------------------------------------------------------------
# hl2 / hlc3 None guard on invalid series input
# ---------------------------------------------------------------------------


class TestHl2Hlc3NoneGuard(TestCase):
    """hl2 and hlc3 must return None when given an invalid/empty series."""

    def test_hl2_returns_none_on_none_high(self):
        """hl2(None, valid) must return None, not raise."""
        result = pandas_ta.hl2(None, pd.Series([1.0, 2.0], dtype=float))
        self.assertIsNone(result)

    def test_hlc3_returns_none_on_none_high(self):
        """hlc3(None, valid, valid) must return None, not raise."""
        v = pd.Series([1.0, 2.0], dtype=float)
        result = pandas_ta.hlc3(None, v, v, talib=False)
        self.assertIsNone(result)

    def test_hl2_valid_input_returns_series(self):
        """hl2 with valid inputs must return a non-empty Series named 'HL2'."""
        h = pd.Series([10.0, 11.0, 12.0])
        low = pd.Series([8.0, 9.0, 10.0])
        result = pandas_ta.hl2(h, low)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "HL2")
        np.testing.assert_allclose(result.values, [9.0, 10.0, 11.0], rtol=1e-9)

    def test_hlc3_valid_input_returns_series(self):
        """hlc3 with valid inputs must return a Series named 'HLC3'."""
        h = pd.Series([12.0, 13.0])
        low = pd.Series([8.0, 9.0])
        c = pd.Series([10.0, 11.0])
        result = pandas_ta.hlc3(h, low, c, talib=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "HLC3")
        np.testing.assert_allclose(result.values, [10.0, 11.0], rtol=1e-9)


# ---------------------------------------------------------------------------
# talib=False propagated to sub-indicator calls
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
        result = pandas_ta.dema(self.close, talib=False)
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
        result = pandas_ta.dema(self.close, length=length, talib=False)
        # drop leading NaNs then compare
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result[mask].values, expected[mask].values, rtol=1e-10)

    def test_tema_talib_false_returns_series(self):
        """tema(talib=False) returns a Series."""
        result = pandas_ta.tema(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_t3_talib_false_returns_series(self):
        """t3(talib=False) returns a Series."""
        result = pandas_ta.t3(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_trima_talib_false_returns_series(self):
        """trima(talib=False) returns a Series."""
        result = pandas_ta.trima(self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_trima_talib_false_matches_manual_calculation(self):
        """trima(talib=False) == sma(sma(close, len1), len2) matching TA-Lib windows."""
        from pandas_ta_classic.overlap.sma import sma

        length = 10
        len1 = length // 2 + 1  # ceil((length+1)/2) — matches TA-Lib
        len2 = (length + 1) // 2  # n/2 for even n, (n+1)/2 for odd n
        sma1 = sma(self.close, length=len1, talib=False)
        expected = sma(sma1, length=len2, talib=False)
        result = pandas_ta.trima(self.close, length=length, talib=False)
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result[mask].values, expected[mask].values, rtol=1e-10)

    def test_cci_talib_false_returns_series(self):
        """cci(talib=False) returns a Series."""
        result = pandas_ta.cci(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, pd.Series)

    def test_natr_talib_false_returns_series(self):
        """natr(talib=False) returns a Series."""
        result = pandas_ta.natr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, pd.Series)


# ---------------------------------------------------------------------------
# ema SMA-seed bounds check
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
        result = pandas_ta.trix(pd.Series(range(1, 31), dtype=float))
        self.assertIsInstance(result, pd.DataFrame)

    def test_chained_ema_over_nan_prefix(self):
        """A NaN prefix from an inner call no longer pushes the seed off the end."""
        close = pd.Series(np.arange(1.0, 201.0))
        result = pandas_ta.ema(pandas_ta.sma(close, length=190), length=20)
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
                func = getattr(pandas_ta, name)
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
        result = pandas_ta.ema(close, length=20)
        self.assertEqual(result.isna().sum(), 19)
        self.assertTrue(np.isfinite(result.iloc[19:]).all())

    def test_bbands_with_zlma_mamode(self):
        """bbands(mamode="zlma") on 5 rows previously raised through ema."""
        result = pandas_ta.bbands(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), mamode="zlma")
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# rma NaN-prefix seed and TA-Lib Wilder seeding for the ADX family
# ---------------------------------------------------------------------------


class TestWilderSeeding(TestCase):
    """rma seeded on iloc[0:length] even when that window started with NaN.

    Chained input (dx, the DM series, true range) always starts with NaN, so
    the seed averaged a partial window and every later value inherited the
    wrong starting level. The ADX family also used that SMA seed where TA-Lib
    uses Wilder's running sum of the first length-1 values: on SPY, ADX_14 was
    up to 49.7% off TA-Lib in its first bars and started 14 bars early.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data().iloc[:2000]

    def test_rma_prefix_result_equals_trimmed_result(self):
        close = self.df["close"].reset_index(drop=True)
        prefixed = close.copy()
        prefixed.iloc[:20] = np.nan
        with_prefix = pandas_ta.rma(prefixed, length=14).to_numpy()[20:]
        without = pandas_ta.rma(close.iloc[20:].reset_index(drop=True), length=14).to_numpy()
        np.testing.assert_array_equal(with_prefix, without)

    def test_wilder_smooth_prefix_result_equals_trimmed_result(self):
        from pandas_ta_classic.utils._wilder import wilder_smooth

        raw = self.df["high"].diff().clip(lower=0).reset_index(drop=True)
        prefixed = raw.copy()
        prefixed.iloc[:20] = np.nan
        with_prefix = wilder_smooth(prefixed, 14).to_numpy()[19:]
        without = wilder_smooth(raw.iloc[19:].reset_index(drop=True), 14).to_numpy()
        np.testing.assert_array_equal(with_prefix, without)

    def test_true_range_undefined_after_nan_prefix(self):
        """The first bar after a NaN prefix has no previous close, like row 0."""
        h, low, c = (self.df[k].reset_index(drop=True).copy() for k in ("high", "low", "close"))
        for s in (h, low, c):
            s.iloc[:20] = np.nan
        tr = pandas_ta.true_range(h, low, c)
        self.assertTrue(tr.iloc[:21].isna().all())
        self.assertTrue(np.isfinite(tr.iloc[21]))
