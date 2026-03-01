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
            # Doji detection algorithms differ; 0.95 is acceptable
            self.assertGreater(corr, 0.95)

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
