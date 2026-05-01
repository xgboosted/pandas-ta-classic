from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import skip, TestCase
from pandas import DataFrame


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

    def test_alma_ext(self):
        self.data.ta.alma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "ALMA_10_6.0_0.85")

    def test_dema_ext(self):
        self.data.ta.dema(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "DEMA_10")

    def test_ema_ext(self):
        self.data.ta.ema(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EMA_10")

    def test_fwma_ext(self):
        self.data.ta.fwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "FWMA_10")

    def test_hilo_ext(self):
        self.data.ta.hilo(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-3:]), ["HILO_13_21", "HILOl_13_21", "HILOs_13_21"]
        )

    def test_hl2_ext(self):
        self.data.ta.hl2(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HL2")

    def test_hlc3_ext(self):
        self.data.ta.hlc3(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HLC3")

    def test_hma_ext(self):
        self.data.ta.hma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HMA_10")

    def test_hwma_ext(self):
        self.data.ta.hwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HWMA_0.2_0.1_0.1")

    def test_jma_ext(self):
        self.data.ta.jma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "JMA_7_0")

    def test_kama_ext(self):
        self.data.ta.kama(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "KAMA_10_2_30")

    def test_ichimoku_ext(self):
        self.data.ta.ichimoku(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-5:]),
            ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
        )

    def test_linreg_ext(self):
        self.data.ta.linreg(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LR_14")

    def test_mcgd_ext(self):
        self.data.ta.mcgd(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MCGD_10")

    def test_midpoint_ext(self):
        self.data.ta.midpoint(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MIDPOINT_2")

    def test_midprice_ext(self):
        self.data.ta.midprice(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MIDPRICE_2")

    def test_ohlc4_ext(self):
        self.data.ta.ohlc4(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "OHLC4")

    def test_pwma_ext(self):
        self.data.ta.pwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PWMA_10")

    def test_rma_ext(self):
        self.data.ta.rma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "RMA_10")

    def test_sinwma_ext(self):
        self.data.ta.sinwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SINWMA_14")

    def test_sma_ext(self):
        self.data.ta.sma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SMA_10")

    def test_ssf_ext(self):
        self.data.ta.ssf(append=True, poles=2)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SSF_10_2")

        self.data.ta.ssf(append=True, poles=3)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SSF_10_3")

    def test_swma_ext(self):
        self.data.ta.swma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SWMA_10")

    def test_supertrend_ext(self):
        self.data.ta.supertrend(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-4:]),
            ["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"],
        )

        result = self.data.ta.supertrend(length=10)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(
            list(result.columns),
            ["SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0"],
        )

        result = self.data.ta.supertrend(multiplier=2.0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(
            list(result.columns),
            ["SUPERT_7_2.0", "SUPERTd_7_2.0", "SUPERTl_7_2.0", "SUPERTs_7_2.0"],
        )

    def test_t3_ext(self):
        self.data.ta.t3(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "T3_10_0.7")

    def test_tema_ext(self):
        self.data.ta.tema(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "TEMA_10")

    def test_trima_ext(self):
        self.data.ta.trima(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "TRIMA_10")

    def test_vidya_ext(self):
        self.data.ta.vidya(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "VIDYA_14")

    def test_vwap_ext(self):
        self.data.ta.vwap(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "VWAP_D")

    def test_vwma_ext(self):
        self.data.ta.vwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "VWMA_10")

    def test_wcp_ext(self):
        self.data.ta.wcp(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "WCP")

    def test_wma_ext(self):
        self.data.ta.wma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "WMA_10")

    def test_zlma_ext(self):
        self.data.ta.zlma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "ZL_EMA_10")

    def test_mmar_ext(self):
        self.data.ta.mmar(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-6:]),
            ["MMAR_10", "MMAR_15", "MMAR_20", "MMAR_25", "MMAR_30", "MMAR_35"],
        )

    def test_rainbow_ext(self):
        self.data.ta.rainbow(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-10:]),
            [
                "RAINBOW_1", "RAINBOW_2", "RAINBOW_3", "RAINBOW_4", "RAINBOW_5",
                "RAINBOW_6", "RAINBOW_7", "RAINBOW_8", "RAINBOW_9", "RAINBOW_10",
            ],
        )

    def test_avgprice_ext(self):
        self.data.ta.avgprice(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "AVGPRICE")

    def test_linregangle_ext(self):
        self.data.ta.linregangle(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LRa_14")

    def test_linregintercept_ext(self):
        self.data.ta.linregintercept(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LRb_14")

    def test_linregslope_ext(self):
        self.data.ta.linregslope(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LRm_14")

    def test_mama_ext(self):
        self.data.ta.mama(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-2:]), ["MAMA_0.5_0.05", "FAMA_0.5_0.05"]
        )

    def test_mavp_ext(self):
        self.data.ta.mavp(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MAVP_2_30")

    def test_medprice_ext(self):
        self.data.ta.medprice(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MEDPRICE")

    def test_tsf_ext(self):
        self.data.ta.tsf(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "TSF_14")

    def test_typprice_ext(self):
        self.data.ta.typprice(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "TYPPRICE")

    def test_ht_trendline_ext(self):
        self.data.ta.ht_trendline(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HT_TRENDLINE")
