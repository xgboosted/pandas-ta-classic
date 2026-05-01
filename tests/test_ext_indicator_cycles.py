from pandas.core.series import Series
from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame


class TestCylesExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dsp_ext(self):
        self.data.ta.dsp(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "DSP_14")

    def test_ebsw_ext(self):
        self.data.ta.ebsw(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EBSW_40_10")

    def test_msw_ext(self):
        self.data.ta.msw(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-2:]), ["MSW_SINE_5", "MSW_LEAD_5"])

    def test_ht_dcperiod_ext(self):
        self.data.ta.ht_dcperiod(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HT_DCPERIOD")

    def test_ht_dcphase_ext(self):
        self.data.ta.ht_dcphase(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HT_DCPHASE")

    def test_ht_phasor_ext(self):
        self.data.ta.ht_phasor(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-2:]), ["HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"]
        )

    def test_ht_sine_ext(self):
        self.data.ta.ht_sine(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-2:]), ["HT_SINE", "HT_LEADSINE"])

    def test_ht_trendmode_ext(self):
        self.data.ta.ht_trendmode(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HT_TRENDMODE")
