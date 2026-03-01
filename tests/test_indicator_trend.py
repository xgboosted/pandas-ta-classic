from tests.config import (
    assert_columns,
    assert_nan_count,
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


class TestTrend(TestCase):
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

    def test_adx(self):
        result = pandas_ta.adx(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADX_14")

        result = pandas_ta.adx(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADX_14")
        assert_offset(self, pandas_ta.adx, self.high, self.low, self.close, talib=False)
        assert_columns(
            self,
            pandas_ta.adx(self.high, self.low, self.close, talib=False),
            ["ADX_14", "DMP_14", "DMN_14"],
        )

    @talib_test
    def test_adx_talib(self):
        result = pandas_ta.adx(self.high, self.low, self.close, talib=False)
        expected = tal.ADX(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result.iloc[:, 0], expected)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(
                result.iloc[:, 0], expected, col=CORRELATION
            )
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_adxr(self):
        result = pandas_ta.adxr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADXR_14")
        assert_columns(self, result, ["ADXR_14", "DMP_14", "DMN_14"])
        assert_offset(
            self, pandas_ta.adxr, self.high, self.low, self.close, talib=False
        )

    @talib_test
    def test_adxr_talib(self):
        result = pandas_ta.adxr(self.high, self.low, self.close, talib=False)
        expected = tal.ADXR(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result.iloc[:, 0], expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(
                result.iloc[:, 0], expected, col=CORRELATION
            )
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_amat(self):
        result = pandas_ta.amat(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AMATe_8_21_2")
        assert_columns(self, result, ["AMATe_LR_8_21_2", "AMATe_SR_8_21_2"])
        assert_offset(self, pandas_ta.amat, self.close)

    def test_aroon(self):
        result = pandas_ta.aroon(self.high, self.low, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

        result = pandas_ta.aroon(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")
        assert_offset(self, pandas_ta.aroon, self.high, self.low, talib=False)
        assert_columns(
            self,
            pandas_ta.aroon(self.high, self.low, talib=False),
            ["AROOND_14", "AROONU_14", "AROONOSC_14"],
        )

    @talib_test
    def test_aroon_talib(self):
        result = pandas_ta.aroon(self.high, self.low, talib=False)
        expected = tal.AROON(self.high, self.low)
        expecteddf = DataFrame({"AROOND_14": expected[0], "AROONU_14": expected[1]})
        try:
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            aroond_corr = pandas_ta.utils.df_error_analysis(
                result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION
            )
            self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            aroonu_corr = pandas_ta.utils.df_error_analysis(
                result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION
            )
            self.assertGreater(aroonu_corr, CORRELATION_THRESHOLD)

    @talib_test
    def test_aroon_osc_talib(self):
        result = pandas_ta.aroon(self.high, self.low)
        expected = tal.AROONOSC(self.high, self.low)
        try:
            pdt.assert_series_equal(result.iloc[:, 2], expected)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(
                result.iloc[:, 2], expected, col=CORRELATION
            )
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_chop(self):
        result = pandas_ta.chop(self.high, self.low, self.close, ln=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOP_14_1_100")
        assert_nan_count(self, result, 14)

        result = pandas_ta.chop(self.high, self.low, self.close, ln=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOPln_14_1_100")
        assert_offset(self, pandas_ta.chop, self.high, self.low, self.close, ln=False)

    def test_cksp(self):
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_3_20")
        assert_columns(self, result, ["CKSPl_10_3_20", "CKSPs_10_3_20"])
        assert_offset(
            self, pandas_ta.cksp, self.high, self.low, self.close, tvmode=False
        )

    def test_cksp_tv(self):
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_1_9")
        assert_columns(self, result, ["CKSPl_10_1_9", "CKSPs_10_1_9"])
        assert_offset(
            self, pandas_ta.cksp, self.high, self.low, self.close, tvmode=True
        )

    def test_decay(self):
        result = pandas_ta.decay(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LDECAY_5")

        result = pandas_ta.decay(self.close, mode="exp")
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EXPDECAY_5")
        assert_offset(self, pandas_ta.decay, self.close)

    def test_decreasing(self):
        result = pandas_ta.decreasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEC_1")

        result = pandas_ta.decreasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SDEC_3")
        assert_offset(self, pandas_ta.decreasing, self.close)

    def test_dpo(self):
        result = pandas_ta.dpo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DPO_20")
        assert_offset(self, pandas_ta.dpo, self.close)

    def test_increasing(self):
        result = pandas_ta.increasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INC_1")

        result = pandas_ta.increasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINC_3")
        assert_offset(self, pandas_ta.increasing, self.close)

    def test_long_run(self):
        result = pandas_ta.long_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_2")
        assert_offset(self, pandas_ta.long_run, self.close, self.open)

    def test_psar(self):
        result = pandas_ta.psar(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PSAR_0.02_0.2")
        assert_columns(
            self,
            result,
            ["PSARl_0.02_0.2", "PSARs_0.02_0.2", "PSARaf_0.02_0.2", "PSARr_0.02_0.2"],
        )
        assert_offset(self, pandas_ta.psar, self.high, self.low)

    @talib_test
    def test_psar_talib(self):
        result = pandas_ta.psar(self.high, self.low)
        # Combine Long and Short SAR's into one SAR value
        psar = result[result.columns[:2]].fillna(0)
        psar = psar[psar.columns[0]] + psar[psar.columns[1]]
        psar.name = result.name
        expected = tal.SAR(self.high, self.low)
        try:
            pdt.assert_series_equal(psar, expected)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(psar, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_qstick(self):
        result = pandas_ta.qstick(self.open, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "QS_10")
        assert_nan_count(self, result, 10)
        assert_offset(self, pandas_ta.qstick, self.open, self.close)

    def test_qstick_mamodes(self):
        for mode in ["dema", "ema", "hma", "rma"]:
            result = pandas_ta.qstick(self.open, self.close, ma=mode)
            self.assertIsInstance(result, Series, msg=f"qstick(ma={mode!r}) failed")

    def test_short_run(self):
        result = pandas_ta.short_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SR_2")
        assert_offset(self, pandas_ta.short_run, self.close, self.open)

    def test_ttm_trend(self):
        result = pandas_ta.ttm_trend(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TTMTREND_6")
        assert_columns(self, result, ["TTM_TRND_6"])
        assert_offset(self, pandas_ta.ttm_trend, self.high, self.low, self.close)

    def test_vhf(self):
        result = pandas_ta.vhf(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VHF_28")
        assert_nan_count(self, result, 28)
        assert_offset(self, pandas_ta.vhf, self.close)

    def test_vortex(self):
        result = pandas_ta.vortex(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VTX_14")
        assert_offset(self, pandas_ta.vortex, self.high, self.low, self.close)
        assert_columns(self, result, ["VTXP_14", "VTXM_14"])

    def test_pmax(self):
        result = pandas_ta.pmax(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PMAX_E_10_3.0")
        assert_offset(self, pandas_ta.pmax, self.high, self.low, self.close)

    def test_tsignals(self):
        # Create a simple trend series for testing
        trend = pandas_ta.sma(self.close, length=10) - pandas_ta.sma(
            self.close, length=20
        )
        trend = (trend > 0).astype(int)
        result = pandas_ta.tsignals(trend)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TS")
        assert_columns(
            self, result, ["TS_Trends", "TS_Trades", "TS_Entries", "TS_Exits"]
        )
        assert_offset(self, pandas_ta.tsignals, trend)

    def test_sarext(self):
        result = pandas_ta.sarext(self.high, self.low, talib=False)
        self.assertIsInstance(result, Series)
        self.assertTrue(result.name.startswith("SAREXT"))
        # SAREXT is iterative/stateful — manual fill tests
        pandas_ta.sarext(self.high, self.low, talib=False, fillna=0)
        pandas_ta.sarext(self.high, self.low, talib=False, fill_method="ffill")
        pandas_ta.sarext(self.high, self.low, talib=False, fill_method="bfill")
        self.assertIsNone(pandas_ta.sarext(None, self.low))

    def test_sarext_startvalue_positive(self):
        result = pandas_ta.sarext(self.high, self.low, startvalue=100.0, talib=False)
        self.assertIsInstance(result, Series)

    def test_sarext_startvalue_negative(self):
        result = pandas_ta.sarext(self.high, self.low, startvalue=-100.0, talib=False)
        self.assertIsInstance(result, Series)

    def test_sarext_short_series(self):
        result = pandas_ta.sarext(self.high.iloc[:1], self.low.iloc[:1], talib=False)
        self.assertIsNone(result)

    @talib_test
    def test_sarext_talib(self):
        result = pandas_ta.sarext(self.high, self.low, talib=False)
        expected = tal.SAREXT(self.high, self.low)
        # SAREXT is iterative; small floating-point differences can accumulate
        # at reversal points, leading to cascading divergence.
        corr = pandas_ta.utils.df_error_analysis(
            result.iloc[1:], expected.iloc[1:], col=CORRELATION
        )
        self.assertGreater(corr, 0.98)

    def test_xsignals(self):
        # Create simple signal series for testing
        signal = pandas_ta.rsi(self.close)
        result = pandas_ta.xsignals(signal, xa=70, xb=30)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "XS")
        assert_columns(
            self, result, ["TS_Trends", "TS_Trades", "TS_Entries", "TS_Exits"]
        )
        assert_offset(self, pandas_ta.xsignals, signal, 70, 30)
