from tests.assertions import assert_indicator_standard, IndicatorSpec
from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
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
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    def test_dsp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dsp,
                args=[self.close],
                expected_name="DSP_14",
                length_override=20,
            ),
        )

    def test_ebsw(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ebsw,
                args=[self.close],
                expected_name="EBSW_40_10",
                length_override=50,
            ),
        )

    def test_ht_dcperiod(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_dcperiod,
                args=[self.close],
                expected_name="HT_DCPERIOD",
            ),
        )

    def test_ht_dcphase(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_dcphase,
                args=[self.close],
                expected_name="HT_DCPHASE",
            ),
        )

    def test_ht_phasor(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_phasor,
                args=[self.close],
                expected_name="HT_PHASOR",
                expected_type=DataFrame,
                expected_columns=["HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"],
            ),
        )

    def test_ht_sine(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_sine,
                args=[self.close],
                expected_name="HT_SINE",
                expected_type=DataFrame,
                expected_columns=["HT_SINE", "HT_LEADSINE"],
            ),
        )

    def test_ht_trendmode(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_trendmode,
                args=[self.close],
                expected_name="HT_TRENDMODE",
            ),
        )

    def test_msw(self):
        result = pandas_ta.msw(self.close, period=10)
        self.assertIsNotNone(result)

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.msw,
                args=[self.close],
                expected_name="MSW_5",
                expected_type=DataFrame,
                expected_columns=["MSW_SINE_5", "MSW_LEAD_5"],
            ),
        )
