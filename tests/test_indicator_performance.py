from tests.assertions import assert_indicator_standard, IndicatorSpec
from tests.config import get_sample_data
import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame


class TestPerformace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.close = cls.data["close"]
        cls.islong = (cls.close > pandas_ta.sma(cls.close, length=8)).astype(int)

    @classmethod
    def tearDownClass(cls):
        del cls.data, cls.close, cls.islong

    def test_log_return(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.log_return,
                args=[self.close],
                expected_name="LOGRET_1",
                length_override=20,
            ),
        )

    def test_cum_log_return(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.log_return,
                args=[self.close],
                expected_name="CUMLOGRET_1",
                kwargs={"cumulative": True},
                length_override=20,
            ),
        )

    def test_percent_return(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.percent_return,
                args=[self.close],
                expected_name="PCTRET_1",
                length_override=20,
            ),
        )

    def test_cum_percent_return(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.percent_return,
                args=[self.close],
                expected_name="CUMPCTRET_1",
                kwargs={"cumulative": True},
                length_override=20,
            ),
        )

    def test_drawdown(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.drawdown,
                args=[self.close],
                expected_name="DD",
                expected_type=DataFrame,
                expected_columns=["DD", "DD_PCT", "DD_LOG"],
            ),
        )
