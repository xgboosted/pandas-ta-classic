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

    def test_ht_dcperiod(self):
        result = pandas_ta.ht_dcperiod(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_DCPERIOD")
        # HT indicators are iterative/stateful — manual fill tests
        pandas_ta.ht_dcperiod(self.close, talib=False, fillna=0)
        pandas_ta.ht_dcperiod(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_dcperiod(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_dcperiod(None))

    @talib_test
    def test_ht_dcperiod_talib(self):
        result = pandas_ta.ht_dcperiod(self.close, talib=False)
        expected = tal.HT_DCPERIOD(self.close)
        corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
        self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ht_dcphase(self):
        result = pandas_ta.ht_dcphase(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_DCPHASE")
        pandas_ta.ht_dcphase(self.close, talib=False, fillna=0)
        pandas_ta.ht_dcphase(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_dcphase(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_dcphase(None))

    @talib_test
    def test_ht_dcphase_talib(self):
        result = pandas_ta.ht_dcphase(self.close, talib=False)
        expected = tal.HT_DCPHASE(self.close)
        corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
        # HT algorithms are recursive; floating-point differences accumulate
        self.assertGreater(corr, 0.90)

    def test_ht_phasor(self):
        result = pandas_ta.ht_phasor(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HT_PHASOR")
        assert_columns(self, result, ["HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"])
        pandas_ta.ht_phasor(self.close, talib=False, fillna=0)
        pandas_ta.ht_phasor(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_phasor(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_phasor(None))

    @talib_test
    def test_ht_phasor_talib(self):
        result = pandas_ta.ht_phasor(self.close, talib=False)
        expected_ip, expected_q = tal.HT_PHASOR(self.close)
        corr_ip = pandas_ta.utils.df_error_analysis(
            result["HT_PHASOR_INPHASE"], expected_ip, col=CORRELATION
        )
        self.assertGreater(corr_ip, CORRELATION_THRESHOLD)
        corr_q = pandas_ta.utils.df_error_analysis(
            result["HT_PHASOR_QUAD"], expected_q, col=CORRELATION
        )
        self.assertGreater(corr_q, CORRELATION_THRESHOLD)

    def test_ht_sine(self):
        result = pandas_ta.ht_sine(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HT_SINE")
        assert_columns(self, result, ["HT_SINE", "HT_LEADSINE"])
        pandas_ta.ht_sine(self.close, talib=False, fillna=0)
        pandas_ta.ht_sine(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_sine(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_sine(None))

    @talib_test
    def test_ht_sine_talib(self):
        result = pandas_ta.ht_sine(self.close, talib=False)
        expected_sine, expected_lead = tal.HT_SINE(self.close)
        corr_sine = pandas_ta.utils.df_error_analysis(
            result["HT_SINE"], expected_sine, col=CORRELATION
        )
        # HT algorithms are recursive; floating-point differences accumulate
        self.assertGreater(corr_sine, 0.90)
        corr_lead = pandas_ta.utils.df_error_analysis(
            result["HT_LEADSINE"], expected_lead, col=CORRELATION
        )
        self.assertGreater(corr_lead, 0.90)

    def test_ht_trendmode(self):
        result = pandas_ta.ht_trendmode(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_TRENDMODE")
        pandas_ta.ht_trendmode(self.close, talib=False, fillna=0)
        pandas_ta.ht_trendmode(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_trendmode(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_trendmode(None))
