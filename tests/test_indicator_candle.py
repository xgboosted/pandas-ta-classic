from tests.assertions import assert_indicator_standard, assert_talib, IndicatorSpec
from tests.config import get_sample_data
import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame, Series

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


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
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ha,
                args=[self.open, self.high, self.low, self.close],
                expected_name="Heikin-Ashi",
                expected_type=DataFrame,
                expected_columns=["HA_open", "HA_high", "HA_low", "HA_close"],
                none_arg_idx=0,
            ),
        )

    def test_cdl_pattern(self):
        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="all")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(len(result.columns), len(pandas_ta.CDL_PATTERN_NAMES))

        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="doji")
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name=["doji", "inside"])
        self.assertIsInstance(result, DataFrame)

    def test_cdl_doji(self):
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.CDLDOJI(self.open, self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_doji,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_DOJI_10_0.1",
                expected_type=Series,
                none_arg_idx=0,
            ),
        )

    def test_cdl_inside(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_inside,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_INSIDE",
                expected_type=Series,
                none_arg_idx=0,
            ),
        )

        result = pandas_ta.cdl_inside(self.open, self.high, self.low, self.close, asbool=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CDL_INSIDE")

    def test_cdl_z(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_z,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_Z_30_1",
                expected_type=DataFrame,
                expected_columns=[
                    "open_Z_30_1",
                    "high_Z_30_1",
                    "low_Z_30_1",
                    "close_Z_30_1",
                ],
                none_arg_idx=0,
            ),
        )
