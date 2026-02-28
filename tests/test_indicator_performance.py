from tests.config import assert_columns, assert_offset, get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame, Series


class TestPerformace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.close = cls.data["close"]
        cls.islong = (cls.close > pandas_ta.sma(cls.close, length=8)).astype(int)
        cls.pctret = pandas_ta.percent_return(cls.close, cumulative=False)
        cls.logret = pandas_ta.percent_return(cls.close, cumulative=False)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.close
        del cls.islong
        del cls.pctret
        del cls.logret

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_log_return(self):
        result = pandas_ta.log_return(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LOGRET_1")
        assert_offset(self, pandas_ta.log_return, self.close)

    def test_cum_log_return(self):
        result = pandas_ta.log_return(self.close, cumulative=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CUMLOGRET_1")
        assert_offset(self, pandas_ta.log_return, self.close, cumulative=True)

    def test_percent_return(self):
        result = pandas_ta.percent_return(self.close, cumulative=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PCTRET_1")
        assert_offset(self, pandas_ta.percent_return, self.close)

    def test_cum_percent_return(self):
        result = pandas_ta.percent_return(self.close, cumulative=True)
        self.assertEqual(result.name, "CUMPCTRET_1")
        assert_offset(self, pandas_ta.percent_return, self.close, cumulative=True)

    def test_drawdown(self):
        result = pandas_ta.drawdown(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DD")
        assert_columns(self, result, ["DD", "DD_PCT", "DD_LOG"])
        assert_offset(self, pandas_ta.drawdown, self.close)
