from tests.config import (
    assert_columns,
    assert_nan_count,
    assert_offset,
    CORRELATION,
    CORRELATION_THRESHOLD,
    get_sample_data,
    HAS_TALIB,
    tal,
    talib_test,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
import pandas as pd
import pandas.testing as pdt
from pandas import DataFrame, Series


class TestOverlap(TestCase):
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

    def test_alma(self):
        result = pandas_ta.alma(
            self.close
        )  # , length=None, sigma=None, distribution_offset=)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ALMA_10_6.0_0.85")
        assert_offset(self, pandas_ta.alma, self.close)

    def test_dema(self):
        result = pandas_ta.dema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEMA_10")

        result = pandas_ta.dema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEMA_10")
        assert_offset(self, pandas_ta.dema, self.close, talib=False)
        assert_nan_count(self, pandas_ta.dema(self.close, talib=False), 10)

    @talib_test
    def test_dema_talib(self):
        result = pandas_ta.dema(self.close, talib=False)
        expected = tal.DEMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ema(self):
        result = pandas_ta.ema(self.close, presma=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        result = pandas_ta.ema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")
        assert_offset(self, pandas_ta.ema, self.close, talib=False)
        assert_nan_count(self, pandas_ta.ema(self.close, talib=False), 10)

    @talib_test
    def test_ema_talib(self):
        result = pandas_ta.ema(self.close, presma=False)
        expected = tal.EMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

        result = pandas_ta.ema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_fwma(self):
        result = pandas_ta.fwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "FWMA_10")
        assert_offset(self, pandas_ta.fwma, self.close)

    def test_hilo(self):
        result = pandas_ta.hilo(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HILO_13_21")
        assert_offset(self, pandas_ta.hilo, self.high, self.low, self.close)
        assert_columns(self, result, ["HILO_13_21", "HILOl_13_21", "HILOs_13_21"])

    def test_hl2(self):
        result = pandas_ta.hl2(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HL2")

        assert_offset(self, pandas_ta.hl2, self.high, self.low)
        assert_nan_count(self, result, 1)

    @talib_test
    def test_hl2_talib(self):
        result = pandas_ta.hl2(self.high, self.low)
        expected = tal.MEDPRICE(self.high, self.low)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_hlc3(self):
        result = pandas_ta.hlc3(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HLC3")

        result = pandas_ta.hlc3(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HLC3")
        assert_offset(self, pandas_ta.hlc3, self.high, self.low, self.close)

    @talib_test
    def test_hlc3_talib(self):
        result = pandas_ta.hlc3(self.high, self.low, self.close, talib=False)
        expected = tal.TYPPRICE(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_hma(self):
        result = pandas_ta.hma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HMA_10")
        assert_offset(self, pandas_ta.hma, self.close)
        assert_nan_count(self, result, 10)

    def test_hwma(self):
        result = pandas_ta.hwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HWMA_0.2_0.1_0.1")
        assert_offset(self, pandas_ta.hwma, self.close)

    def test_kama(self):
        result = pandas_ta.kama(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "KAMA_10_2_30")

        assert_offset(self, pandas_ta.kama, self.close)
        assert_nan_count(self, result, 10)

    @talib_test
    def test_kama_talib(self):
        result = pandas_ta.kama(self.close)
        expected = tal.KAMA(self.close, timeperiod=10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_jma(self):
        result = pandas_ta.jma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "JMA_7_0")
        assert_offset(self, pandas_ta.jma, self.close)

    def test_ichimoku(self):
        ichimoku, span = pandas_ta.ichimoku(self.high, self.low, self.close)
        self.assertIsInstance(ichimoku, DataFrame)
        self.assertIsInstance(span, DataFrame)
        self.assertEqual(ichimoku.name, "ICHIMOKU_9_26_52")
        self.assertEqual(span.name, "ICHISPAN_9_26")
        assert_columns(self, ichimoku, ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"])
        assert_columns(self, span, ["ISA_9", "ISB_26"])

        # include_chikou=False omits the chikou_span column (line 108-109)
        ichi_no_chikou, _ = pandas_ta.ichimoku(
            self.high, self.low, self.close, include_chikou=False
        )
        assert_columns(self, ichi_no_chikou, ["ISA_9", "ISB_26", "ITS_9", "IKS_26"])

        # lookahead=False sets include_chikou=False internally (line 29-30)
        ichi_no_look, _ = pandas_ta.ichimoku(
            self.high, self.low, self.close, lookahead=False
        )
        assert_columns(self, ichi_no_look, ["ISA_9", "ISB_26", "ITS_9", "IKS_26"])

        # offset shifts all series (lines 51-55)
        ichi_off, _ = pandas_ta.ichimoku(self.high, self.low, self.close, offset=1)
        self.assertIsInstance(ichi_off, DataFrame)

        # fillna and fill_method cover lines 58-89
        pandas_ta.ichimoku(self.high, self.low, self.close, fillna=0)
        pandas_ta.ichimoku(self.high, self.low, self.close, fill_method="ffill")
        pandas_ta.ichimoku(self.high, self.low, self.close, fill_method="bfill")

        # None-guard returns (None, None) tuple (line 32-33)
        result_a, result_b = pandas_ta.ichimoku(None, self.low, self.close)
        self.assertIsNone(result_a)
        self.assertIsNone(result_b)

    def test_linreg(self):
        result = pandas_ta.linreg(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_14")

        result = pandas_ta.linreg(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_14")
        assert_offset(self, pandas_ta.linreg, self.close, talib=False)

    @talib_test
    def test_linreg_talib(self):
        result = pandas_ta.linreg(self.close, talib=False)
        expected = tal.LINEARREG(self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_linreg_angle(self):
        result = pandas_ta.linreg(self.close, angle=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRa_14")

        result = pandas_ta.linreg(self.close, angle=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRa_14")
        assert_offset(self, pandas_ta.linreg, self.close, angle=True, talib=False)

    @talib_test
    def test_linreg_angle_talib(self):
        result = pandas_ta.linreg(self.close, angle=True, talib=False)
        expected = tal.LINEARREG_ANGLE(self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_linreg_intercept(self):
        result = pandas_ta.linreg(self.close, intercept=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRb_14")

        result = pandas_ta.linreg(self.close, intercept=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRb_14")
        assert_offset(self, pandas_ta.linreg, self.close, intercept=True, talib=False)

    @talib_test
    def test_linreg_intercept_talib(self):
        result = pandas_ta.linreg(self.close, intercept=True, talib=False)
        expected = tal.LINEARREG_INTERCEPT(self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_linreg_r(self):
        result = pandas_ta.linreg(self.close, r=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRr_14")
        assert_offset(self, pandas_ta.linreg, self.close, r=True)

    def test_linreg_slope(self):
        result = pandas_ta.linreg(self.close, slope=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRm_14")

        result = pandas_ta.linreg(self.close, slope=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRm_14")
        assert_offset(self, pandas_ta.linreg, self.close, slope=True, talib=False)

    @talib_test
    def test_linreg_slope_talib(self):
        result = pandas_ta.linreg(self.close, slope=True, talib=False)
        expected = tal.LINEARREG_SLOPE(self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ma(self):
        result = pandas_ta.ma()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        result = pandas_ta.ma("ema", self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        result = pandas_ta.ma("fwma", self.close, length=15)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "FWMA_15")

    def test_mcgd(self):
        result = pandas_ta.mcgd(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MCGD_10")
        assert_offset(self, pandas_ta.mcgd, self.close)

    def test_midpoint(self):
        result = pandas_ta.midpoint(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPOINT_2")

        result = pandas_ta.midpoint(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPOINT_2")
        assert_offset(self, pandas_ta.midpoint, self.close, talib=False)

    @talib_test
    def test_midpoint_talib(self):
        result = pandas_ta.midpoint(self.close, talib=False)
        expected = tal.MIDPOINT(self.close, 2)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_midprice(self):
        result = pandas_ta.midprice(self.high, self.low, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPRICE_2")

        result = pandas_ta.midprice(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPRICE_2")
        assert_offset(self, pandas_ta.midprice, self.high, self.low, talib=False)

    @talib_test
    def test_midprice_talib(self):
        result = pandas_ta.midprice(self.high, self.low, talib=False)
        expected = tal.MIDPRICE(self.high, self.low, 2)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ohlc4(self):
        result = pandas_ta.ohlc4(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OHLC4")

        assert_offset(self, pandas_ta.ohlc4, self.open, self.high, self.low, self.close)
        assert_nan_count(self, result, 1)

    @talib_test
    def test_ohlc4_talib(self):
        result = pandas_ta.ohlc4(self.open, self.high, self.low, self.close)
        expected = tal.AVGPRICE(self.open, self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_pwma(self):
        result = pandas_ta.pwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PWMA_10")
        assert_offset(self, pandas_ta.pwma, self.close)

    def test_rma(self):
        result = pandas_ta.rma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RMA_10")
        assert_offset(self, pandas_ta.rma, self.close)

    def test_sinwma(self):
        result = pandas_ta.sinwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINWMA_14")
        assert_offset(self, pandas_ta.sinwma, self.close)

    def test_sma(self):
        result = pandas_ta.sma(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SMA_10")

        result = pandas_ta.sma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SMA_10")
        sma_result = pandas_ta.sma(self.close, talib=False)
        pd.testing.assert_series_equal(
            sma_result, self.close.rolling(10).mean(), check_names=False
        )
        assert_offset(self, pandas_ta.sma, self.close, talib=False)
        assert_nan_count(self, sma_result, 10)

    @talib_test
    def test_sma_talib(self):
        result = pandas_ta.sma(self.close, talib=False)
        expected = tal.SMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ssf(self):
        result = pandas_ta.ssf(self.close, poles=2)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_2")

        result = pandas_ta.ssf(self.close, poles=3)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_3")
        assert_offset(self, pandas_ta.ssf, self.close)

    def test_swma(self):
        result = pandas_ta.swma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SWMA_10")
        assert_offset(self, pandas_ta.swma, self.close)

    def test_supertrend(self):
        result = pandas_ta.supertrend(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SUPERT_7_3.0")
        assert_offset(self, pandas_ta.supertrend, self.high, self.low, self.close)
        assert_columns(
            self,
            result,
            ["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"],
        )

    def test_t3(self):
        result = pandas_ta.t3(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "T3_10_0.7")

        result = pandas_ta.t3(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "T3_10_0.7")
        assert_offset(self, pandas_ta.t3, self.close, talib=False)

    @talib_test
    def test_t3_talib(self):
        result = pandas_ta.t3(self.close, talib=False)
        expected = tal.T3(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_tema(self):
        result = pandas_ta.tema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TEMA_10")

        result = pandas_ta.tema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TEMA_10")
        assert_offset(self, pandas_ta.tema, self.close, talib=False)
        assert_nan_count(self, pandas_ta.tema(self.close, talib=False), 10)

    @talib_test
    def test_tema_talib(self):
        result = pandas_ta.tema(self.close, talib=False)
        expected = tal.TEMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_trima(self):
        result = pandas_ta.trima(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRIMA_10")

        result = pandas_ta.trima(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRIMA_10")
        assert_offset(self, pandas_ta.trima, self.close, talib=False)

    @talib_test
    def test_trima_talib(self):
        result = pandas_ta.trima(self.close, talib=False)
        expected = tal.TRIMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_vidya(self):
        result = pandas_ta.vidya(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VIDYA_14")
        assert_offset(self, pandas_ta.vidya, self.close)

    def test_vwap(self):
        result = pandas_ta.vwap(self.high, self.low, self.close, self.volume)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VWAP_D")
        assert_offset(
            self,
            pandas_ta.vwap,
            self.high,
            self.low,
            self.close,
            self.volume,
            none_arg_idx=None,
        )

    def test_vwma(self):
        result = pandas_ta.vwma(self.close, self.volume)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VWMA_10")
        assert_offset(self, pandas_ta.vwma, self.close, self.volume)

    def test_wcp(self):
        result = pandas_ta.wcp(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WCP")

        result = pandas_ta.wcp(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WCP")
        assert_offset(self, pandas_ta.wcp, self.high, self.low, self.close)

    @talib_test
    def test_wcp_talib(self):
        result = pandas_ta.wcp(self.high, self.low, self.close, talib=False)
        expected = tal.WCLPRICE(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_wma(self):
        result = pandas_ta.wma(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WMA_10")

        result = pandas_ta.wma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WMA_10")
        assert_offset(self, pandas_ta.wma, self.close, talib=False)
        assert_nan_count(self, pandas_ta.wma(self.close, talib=False), 10)

    @talib_test
    def test_wma_talib(self):
        result = pandas_ta.wma(self.close, talib=False)
        expected = tal.WMA(self.close, 10)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_zlma(self):
        result = pandas_ta.zlma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ZL_EMA_10")
        assert_offset(self, pandas_ta.zlma, self.close)

    def test_mmar(self):
        result = pandas_ta.mmar(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MMAR_10_5_6")
        assert_offset(self, pandas_ta.mmar, self.close)

    def test_rainbow(self):
        result = pandas_ta.rainbow(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "RAINBOW_2_10")
        assert_offset(self, pandas_ta.rainbow, self.close)

    def test_tsf(self):
        result = pandas_ta.tsf(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TSF_14")
        assert_offset(self, pandas_ta.tsf, self.close, talib=False)

    @talib_test
    def test_tsf_talib(self):
        result = pandas_ta.tsf(self.close, talib=False)
        expected = tal.TSF(self.close, timeperiod=14)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ht_trendline(self):
        result = pandas_ta.ht_trendline(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HT_TRENDLINE")
        # HT indicators are iterative/stateful — manual fill tests
        pandas_ta.ht_trendline(self.close, talib=False, fillna=0)
        pandas_ta.ht_trendline(self.close, talib=False, fill_method="ffill")
        pandas_ta.ht_trendline(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.ht_trendline(None))

    @talib_test
    def test_ht_trendline_talib(self):
        result = pandas_ta.ht_trendline(self.close, talib=False)
        expected = tal.HT_TRENDLINE(self.close)
        corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
        self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_mama(self):
        result = pandas_ta.mama(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MAMA_0.5_0.05")
        assert_columns(self, result, ["MAMA_0.5_0.05", "FAMA_0.5_0.05"])
        # MAMA is iterative/stateful — manual fill tests
        pandas_ta.mama(self.close, talib=False, fillna=0)
        pandas_ta.mama(self.close, talib=False, fill_method="ffill")
        pandas_ta.mama(self.close, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.mama(None))

    @talib_test
    def test_mama_talib(self):
        result = pandas_ta.mama(self.close, talib=False)
        expected_mama, expected_fama = tal.MAMA(self.close)
        corr_mama = pandas_ta.utils.df_error_analysis(
            result.iloc[:, 0], expected_mama, col=CORRELATION
        )
        self.assertGreater(corr_mama, CORRELATION_THRESHOLD)
        corr_fama = pandas_ta.utils.df_error_analysis(
            result.iloc[:, 1], expected_fama, col=CORRELATION
        )
        self.assertGreater(corr_fama, CORRELATION_THRESHOLD)
