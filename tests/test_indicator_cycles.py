from tests.config import (
    error_analysis,
    get_sample_data,
    CORRELATION,
    CORRELATION_THRESHOLD,
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

    def test_ebsw(self):
        result = pandas_ta.ebsw(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EBSW_40_10")

    def test_ht_dcperiod(self):
        result = pandas_ta.ht_dcperiod(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_DCPERIOD")
        pandas_ta.ht_dcperiod(self.close, fillna=0)
        pandas_ta.ht_dcperiod(self.close, fill_method="ffill")
        pandas_ta.ht_dcperiod(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_dcperiod(None))

    def test_ht_dcphase(self):
        result = pandas_ta.ht_dcphase(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_DCPHASE")
        pandas_ta.ht_dcphase(self.close, fillna=0)
        pandas_ta.ht_dcphase(self.close, fill_method="ffill")
        pandas_ta.ht_dcphase(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_dcphase(None))

    def test_ht_phasor(self):
        result = pandas_ta.ht_phasor(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HT_PHASOR")
        self.assertListEqual(
            list(result.columns), ["HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"]
        )
        pandas_ta.ht_phasor(self.close, fillna=0)
        pandas_ta.ht_phasor(self.close, fill_method="ffill")
        pandas_ta.ht_phasor(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_phasor(None))

    def test_ht_sine(self):
        result = pandas_ta.ht_sine(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HT_SINE")
        self.assertListEqual(list(result.columns), ["HT_SINE", "HT_LEADSINE"])
        pandas_ta.ht_sine(self.close, fillna=0)
        pandas_ta.ht_sine(self.close, fill_method="ffill")
        pandas_ta.ht_sine(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_sine(None))

    def test_ht_trendmode(self):
        result = pandas_ta.ht_trendmode(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_TRENDMODE")
        pandas_ta.ht_trendmode(self.close, fillna=0)
        pandas_ta.ht_trendmode(self.close, fill_method="ffill")
        pandas_ta.ht_trendmode(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_trendmode(None))

    def test_msw(self):
        result = pandas_ta.msw(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MSW_5")
        self.assertListEqual(list(result.columns), ["MSW_SINE_5", "MSW_LEAD_5"])
        result_p10 = pandas_ta.msw(self.close, period=10)
        self.assertEqual(result_p10.name, "MSW_10")
        self.assertListEqual(list(result_p10.columns), ["MSW_SINE_10", "MSW_LEAD_10"])
        pandas_ta.msw(self.close, fillna=0)
        pandas_ta.msw(self.close, fill_method="ffill")
        pandas_ta.msw(self.close, fill_method="bfill")
        self.assertIsNone(pandas_ta.msw(None))
