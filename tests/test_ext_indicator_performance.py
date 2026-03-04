from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame


class TestPerformaceExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_data = get_sample_data()
        cls.islong = cls._original_data["close"] > pandas_ta.sma(
            cls._original_data["close"], length=50
        )

    @classmethod
    def tearDownClass(cls):
        del cls._original_data
        del cls.islong

    def setUp(self):
        self.data = self._original_data.copy()

    def test_log_return_ext(self):
        self.data.ta.log_return(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LOGRET_1")

    def test_cum_log_return_ext(self):
        self.data.ta.log_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMLOGRET_1")

    def test_percent_return_ext(self):
        self.data.ta.percent_return(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PCTRET_1")

    def test_cum_percent_return_ext(self):
        self.data.ta.percent_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMPCTRET_1")

    def test_drawdown_ext(self):
        self.data.ta.drawdown(append=True)
        self.assertIsInstance(self.data, DataFrame)
        # Drawdown returns 3 columns: DD, DD_PCT, DD_LOG
        self.assertEqual(list(self.data.columns[-3:]), ["DD", "DD_PCT", "DD_LOG"])
