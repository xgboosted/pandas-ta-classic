from tests.config import (
    assert_offset,
    get_sample_data,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase, skip
import pandas.testing as pdt
from pandas import DataFrame, Series

try:
    import talib as tal

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    tal = None


class TestCycles(TestCase):
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

    def test_dsp(self):
        result = pandas_ta.dsp(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DSP_14")
        assert_offset(self, pandas_ta.dsp, self.close)

    def test_ebsw(self):
        result = pandas_ta.ebsw(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EBSW_40_10")
        assert_offset(self, pandas_ta.ebsw, self.close)
