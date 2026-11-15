import warnings
from unittest import TestCase

from pandas import DataFrame

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase
from tests.config import get_sample_data


class TestOverlapExtension(TestCase):
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

    def test_ichimoku_append_span_ext(self):
        from tests.config import get_sample_data

        data = get_sample_data()
        # append_span is a functional flag forwarded to the underlying
        # function; it emits no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data.ta.ichimoku(append=True, append_span=True)
        self.assertIsInstance(data, DataFrame)
        # Span columns (ISA_9, ISB_26) should be present
        self.assertIn("ISA_9", data.columns)
        self.assertIn("ISB_26", data.columns)
        # Main ichimoku columns should also be present
        self.assertIn("ITS_9", data.columns)
        self.assertIn("IKS_26", data.columns)

    def test_ichimoku_accessor_no_warning(self):
        from tests.config import get_sample_data

        data = get_sample_data()
        # 0.9.0 ships no deprecation: the accessor call emits no warning at all.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data.ta.ichimoku(append=True)
        self.assertIsInstance(data, DataFrame)

    def test_ichimoku_prefix_applied_once(self):
        from tests.config import get_sample_data

        data = get_sample_data()
        base = list(data.columns)
        result = data.ta.ichimoku(prefix="P", suffix="S", append=True)
        expected = ["P_ISA_9_S", "P_ISB_26_S", "P_ITS_9_S", "P_IKS_26_S", "P_ICS_26_S"]
        self.assertEqual(list(result.columns), expected)
        self.assertEqual(list(data.columns), base + expected)


class TestExtOverlapDeeper(AccessorValueCase):
    def test_ema_ext(self):
        df = self.fresh()
        df.ta.ema(append=True)
        self._check_appended(df, "EMA_10")

    def test_fwma_ext(self):
        df = self.fresh()
        df.ta.fwma(append=True)
        self._check_appended(df, "FWMA_10")

    def test_hma_ext(self):
        df = self.fresh()
        df.ta.hma(append=True)
        self._check_appended(df, "HMA_10")

    def test_hwc_ext(self):
        df = self.fresh()
        df.ta.hwc(append=True)
        self._check_appended(df, "HWM")

    def test_kama_ext(self):
        df = self.fresh()
        df.ta.kama(append=True)
        self._check_appended(df, "KAMA_10_2_30")

    def test_linreg_ext(self):
        df = self.fresh()
        df.ta.linreg(append=True)
        self._check_appended(df, "LR_14")

    def test_rma_ext(self):
        df = self.fresh()
        df.ta.rma(append=True)
        self._check_appended(df, "RMA_10")

    def test_sma_ext(self):
        df = self.fresh()
        df.ta.sma(append=True)
        self._check_appended(df, "SMA_10")

    def test_ssf_ext(self):
        df = self.fresh()
        df.ta.ssf(append=True)
        self._check_appended(df, "SSF_10_2")

    def test_vwap_ext(self):
        df = self.fresh()
        df.ta.vwap(append=True)
        self._check_appended(df, "VWAP_D")

    def test_vwma_ext(self):
        df = self.fresh()
        df.ta.vwma(append=True)
        self._check_appended(df, "VWMA_10")

    def test_zlma_ext(self):
        df = self.fresh()
        df.ta.zlma(append=True)
        # ZL_EMA_10 is the default; just check the last appended column
        self._check_last_col(df)

    def test_accbands_ext(self):
        df = self.fresh()
        df.ta.accbands(append=True)
        self._check_appended(df, "ACCBL_20")
        self._check_appended(df, "ACCBM_20")
        self._check_appended(df, "ACCBU_20")

    def test_bbands_ext(self):
        df = self.fresh()
        df.ta.bbands(append=True)
        bb_cols = [c for c in df.columns if c.startswith("BB")]
        self.assertTrue(len(bb_cols) >= 3, "Expected at least 3 BBands columns")
        for col in bb_cols:
            self._check_appended(df, col)
