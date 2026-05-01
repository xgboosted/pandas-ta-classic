from tests.config import get_sample_data
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

    def test_cum_log_return(self):
        result = pandas_ta.log_return(self.close, cumulative=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CUMLOGRET_1")

    def test_percent_return(self):
        result = pandas_ta.percent_return(self.close, cumulative=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PCTRET_1")

    def test_cum_percent_return(self):
        result = pandas_ta.percent_return(self.close, cumulative=True)
        self.assertEqual(result.name, "CUMPCTRET_1")

    def test_drawdown(self):
        result = pandas_ta.drawdown(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DD")

        result = pandas_ta.drawdown(self.close, offset=1)
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.drawdown(self.close, fillna=0)
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.drawdown(self.close, fill_method="ffill")
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.drawdown(self.close, fill_method="bfill")
        self.assertIsInstance(result, DataFrame)
