from tests.config import (
    assert_columns,
    assert_offset,
    get_sample_data,
    CORRELATION,
    CORRELATION_THRESHOLD,
    HAS_TALIB,
    tal,
    talib_test,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase, skip
import pandas.testing as pdt
from pandas import DataFrame, Series


class TestCandle(TestCase):
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

    def test_ha(self):
        result = pandas_ta.ha(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "Heikin-Ashi")
        assert_offset(
            self,
            pandas_ta.ha,
            self.open,
            self.high,
            self.low,
            self.close,
            expected_cols=["HA_open", "HA_high", "HA_low", "HA_close"],
        )

    def test_cdl_pattern(self):
        result = pandas_ta.cdl_pattern(
            self.open, self.high, self.low, self.close, name="all"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(len(result.columns), len(pandas_ta.CDL_PATTERN_NAMES))

        result = pandas_ta.cdl_pattern(
            self.open, self.high, self.low, self.close, name="doji"
        )
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.cdl_pattern(
            self.open, self.high, self.low, self.close, name=["doji", "inside"]
        )
        self.assertIsInstance(result, DataFrame)

    def test_cdl_doji(self):
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CDL_DOJI_10_0.1")
        assert_offset(
            self, pandas_ta.cdl_doji, self.open, self.high, self.low, self.close
        )

    @talib_test
    def test_cdl_doji_talib(self):
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close)
        expected = tal.CDLDOJI(self.open, self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(
                result, expected, check_names=False, check_dtype=False
            )
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_cdl_inside(self):
        result = pandas_ta.cdl_inside(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CDL_INSIDE")

        result = pandas_ta.cdl_inside(
            self.open, self.high, self.low, self.close, asbool=True
        )
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CDL_INSIDE")
        assert_offset(
            self, pandas_ta.cdl_inside, self.open, self.high, self.low, self.close
        )

    def test_cdl_z(self):
        result = pandas_ta.cdl_z(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CDL_Z_30_1")
        assert_offset(
            self,
            pandas_ta.cdl_z,
            self.open,
            self.high,
            self.low,
            self.close,
            expected_cols=["open_Z_30_1", "high_Z_30_1", "low_Z_30_1", "close_Z_30_1"],
        )

    def test_cdl_z_full(self):
        result = pandas_ta.cdl_z(self.open, self.high, self.low, self.close, full=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CDL_Za")
        assert_columns(self, result, ["open_Za", "high_Za", "low_Za", "close_Za"])
        # full=True uses bfill, so no NaNs should remain
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_cdl_pattern_invalid_name(self):
        result = pandas_ta.cdl_pattern(
            self.open, self.high, self.low, self.close, name="nonexistent"
        )
        self.assertIsNone(result)

    # -- Native pattern offset tests --
    def test_cdl_hammer(self):
        result = pandas_ta.cdl_hammer(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        assert_offset(
            self, pandas_ta.cdl_hammer, self.open, self.high, self.low, self.close
        )

    def test_cdl_engulfing(self):
        result = pandas_ta.cdl_engulfing(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        assert_offset(
            self, pandas_ta.cdl_engulfing, self.open, self.high, self.low, self.close
        )

    def test_cdl_morningstar(self):
        result = pandas_ta.cdl_morningstar(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        assert_offset(
            self, pandas_ta.cdl_morningstar, self.open, self.high, self.low, self.close
        )

    def test_cdl_3inside(self):
        result = pandas_ta.cdl_3inside(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        assert_offset(
            self, pandas_ta.cdl_3inside, self.open, self.high, self.low, self.close
        )

    def test_cdl_hikkake(self):
        result = pandas_ta.cdl_hikkake(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        assert_offset(
            self, pandas_ta.cdl_hikkake, self.open, self.high, self.low, self.close
        )

    def test_cdl_risefall3methods(self):
        result = pandas_ta.cdl_risefall3methods(
            self.open, self.high, self.low, self.close
        )
        self.assertIsInstance(result, Series)
        assert_offset(
            self,
            pandas_ta.cdl_risefall3methods,
            self.open,
            self.high,
            self.low,
            self.close,
        )

    # -- TA-Lib cross-validation: all native patterns at once --
    @talib_test
    def test_all_native_patterns_talib(self):
        """Cross-validate every native pattern against TA-Lib (exact equality)."""
        from pandas_ta_classic.candles.cdl_pattern import _NATIVE_PATTERNS

        import talib

        for name, func in sorted(_NATIVE_PATTERNS.items()):
            talib_name = f"CDL{name.upper()}"
            talib_func = getattr(talib, talib_name, None)
            if talib_func is None:
                continue
            with self.subTest(pattern=name):
                native = func(self.open, self.high, self.low, self.close)
                expected = Series(
                    talib_func(self.open, self.high, self.low, self.close).astype(
                        float
                    ),
                    index=self.close.index,
                )
                pdt.assert_series_equal(
                    native, expected, check_names=False, check_dtype=False
                )
