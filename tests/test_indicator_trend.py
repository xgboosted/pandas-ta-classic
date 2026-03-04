from tests.config import (
    error_analysis,
    get_sample_data,
    CORRELATION,
    CORRELATION_THRESHOLD,
    VERBOSE,
)
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase, skip
import pandas.testing as pdt
from pandas import DataFrame, Series

try:
    import talib as tal

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    tal = None


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

        try:
            expected = tal.ADX(self.high, self.low, self.close)
            pdt.assert_series_equal(result.iloc[:, 0], expected)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.adx(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADX_14")

    def test_amat(self):
        result = pandas_ta.amat(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AMATe_8_21_2")

    def test_aroon(self):
        result = pandas_ta.aroon(self.high, self.low, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

        try:
            expected = tal.AROON(self.high, self.low)
            expecteddf = DataFrame({"AROOND_14": expected[0], "AROONU_14": expected[1]})
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                aroond_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION
                )
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                aroonu_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION
                )
                self.assertGreater(aroonu_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

        result = pandas_ta.aroon(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

    def test_aroon_osc(self):
        result = pandas_ta.aroon(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

        try:
            expected = tal.AROONOSC(self.high, self.low)
            pdt.assert_series_equal(result.iloc[:, 2], expected)
        except AssertionError:
            try:
                aroond_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 2], expected, col=CORRELATION
                )
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

    def test_chop(self):
        result = pandas_ta.chop(self.high, self.low, self.close, ln=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOP_14_1_100")

        result = pandas_ta.chop(self.high, self.low, self.close, ln=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOPln_14_1_100")

    def test_cksp(self):
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_3_20")

    def test_cksp_tv(self):
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_1_9")

    def test_cpr_basic(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, levels="basic"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CPR")
        self.assertIn("CPR_TC", result.columns)
        self.assertIn("CPR_PIVOT", result.columns)
        self.assertIn("CPR_BC", result.columns)

    def test_cpr_classic_standard(self):
        result = pandas_ta.cpr(
            self.open,
            self.high,
            self.low,
            self.close,
            method="classic",
            levels="standard",
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R1", result.columns)
        self.assertIn("CPR_R2", result.columns)
        self.assertIn("CPR_S1", result.columns)
        self.assertIn("CPR_S2", result.columns)

    def test_cpr_classic_extended(self):
        result = pandas_ta.cpr(
            self.open,
            self.high,
            self.low,
            self.close,
            method="classic",
            levels="extended",
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R3", result.columns)
        self.assertIn("CPR_R4", result.columns)
        self.assertIn("CPR_S3", result.columns)
        self.assertIn("CPR_S4", result.columns)

    def test_cpr_camarilla(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, method="camarilla", levels="all"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R4", result.columns)
        self.assertIn("CPR_S4", result.columns)

    def test_cpr_fibonacci(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, method="fibonacci", levels="all"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R3", result.columns)
        self.assertIn("CPR_S3", result.columns)

    def test_cpr_woodie(self):
        result = pandas_ta.cpr(
            self.open,
            self.high,
            self.low,
            self.close,
            method="woodie",
            levels="standard",
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R2", result.columns)
        self.assertIn("CPR_S2", result.columns)

    def test_cpr_width_analysis(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, width_analysis=True
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_WIDTH", result.columns)
        self.assertIn("CPR_WIDTH_PCT", result.columns)
        self.assertIn("CPR_WIDTH_CLASS", result.columns)

    def test_cpr_price_position(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, price_position=True
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_POSITION", result.columns)

    def test_cpr_option_basic(self):
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, strike_round=50
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CPR_OPTION")
        # Check base CPR columns
        self.assertIn("CPR_TC", result.columns)
        self.assertIn("CPR_PIVOT", result.columns)
        self.assertIn("CPR_BC", result.columns)
        # Check option strike columns
        self.assertIn("OPT_TC_STRIKE", result.columns)
        self.assertIn("OPT_PIVOT_STRIKE", result.columns)
        self.assertIn("OPT_BC_STRIKE", result.columns)
        self.assertIn("OPT_CALL_OTM1", result.columns)
        self.assertIn("OPT_CALL_OTM2", result.columns)
        self.assertIn("OPT_PUT_OTM1", result.columns)
        self.assertIn("OPT_PUT_OTM2", result.columns)

    def test_cpr_option_signals(self):
        result = pandas_ta.cpr_option(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        # Check signal columns
        self.assertIn("OPT_BREAKOUT_CALL", result.columns)
        self.assertIn("OPT_BREAKOUT_PUT", result.columns)
        self.assertIn("OPT_REJECTION_CALL", result.columns)
        self.assertIn("OPT_REJECTION_PUT", result.columns)

    def test_cpr_option_expected_range(self):
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, volatility_factor=2.0
        )
        self.assertIsInstance(result, DataFrame)
        # Check expected range columns
        self.assertIn("OPT_RANGE_HIGH", result.columns)
        self.assertIn("OPT_RANGE_LOW", result.columns)

    def test_cpr_option_strike_rounding(self):
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, strike_round=100
        )
        self.assertIsInstance(result, DataFrame)
        # Check that strikes are rounded to 100
        tc_strike = result["OPT_TC_STRIKE"].dropna()
        if len(tc_strike) > 0:
            # All strikes should be multiples of 100
            self.assertTrue(all(tc_strike % 100 == 0))

    def test_cpr_option_camarilla(self):
        result = pandas_ta.cpr_option(
            self.open,
            self.high,
            self.low,
            self.close,
            method="camarilla",
            levels="all",
            strike_round=50,
        )
        self.assertIsInstance(result, DataFrame)
        # Should have all Camarilla levels plus option columns
        self.assertIn("CPR_R4", result.columns)
        self.assertIn("CPR_S4", result.columns)
        self.assertIn("OPT_CALL_OTM1", result.columns)

    def test_cpr_option_custom_parameters(self):
        result = pandas_ta.cpr_option(
            self.open,
            self.high,
            self.low,
            self.close,
            method="fibonacci",
            levels="all",
            strike_round=25,
            volatility_factor=1.2,
            breakout_lookback=3,
            rejection_threshold=0.5,
        )
        self.assertIsInstance(result, DataFrame)
        # Check all expected columns exist
        self.assertIn("OPT_RANGE_HIGH", result.columns)
        self.assertIn("OPT_RANGE_LOW", result.columns)
        self.assertIn("OPT_BREAKOUT_CALL", result.columns)

    def test_cpr_virgin_detection(self):
        result = pandas_ta.cpr(
            self.open,
            self.high,
            self.low,
            self.close,
            virgin_cpr=True,
            virgin_lookforward=5,
        )
        self.assertIsInstance(result, DataFrame)
        # Check virgin CPR column exists
        self.assertIn("CPR_VIRGIN", result.columns)
        # Virgin column should be boolean type
        virgin_values = result["CPR_VIRGIN"].dropna()
        if len(virgin_values) > 0:
            self.assertTrue(virgin_values.dtype == bool)

    def test_cpr_virgin_disabled(self):
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, virgin_cpr=False
        )
        self.assertIsInstance(result, DataFrame)
        # Virgin column should NOT exist when disabled
        self.assertNotIn("CPR_VIRGIN", result.columns)

    def test_cpr_option_virgin_default(self):
        result = pandas_ta.cpr_option(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        # Virgin CPR should be enabled by default in cpr_option
        self.assertIn("CPR_VIRGIN", result.columns)

    def test_cpr_option_virgin_disabled(self):
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, virgin_cpr=False
        )
        self.assertIsInstance(result, DataFrame)
        # Virgin column should NOT exist when explicitly disabled
        self.assertNotIn("CPR_VIRGIN", result.columns)

    def test_cpr_virgin_custom_lookforward(self):
        result = pandas_ta.cpr(
            self.open,
            self.high,
            self.low,
            self.close,
            virgin_cpr=True,
            virgin_lookforward=10,
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_VIRGIN", result.columns)

    def test_cpr_invalid_method(self):
        # Test with invalid method - should default to 'classic'
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, method="invalid_method"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_TC", result.columns)

    def test_cpr_invalid_timeframe(self):
        # Test with invalid timeframe - should default to 'daily'
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, timeframe="invalid_timeframe"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_TC", result.columns)

    def test_cpr_invalid_levels(self):
        # Test with invalid levels - should default to 'standard'
        result = pandas_ta.cpr(
            self.open, self.high, self.low, self.close, levels="invalid_levels"
        )
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_R1", result.columns)

    def test_cpr_empty_series(self):
        # Test with empty series - should return None
        empty_series = Series(dtype=float)
        result = pandas_ta.cpr(empty_series, empty_series, empty_series, empty_series)
        self.assertIsNone(result)

    def test_cpr_with_nans(self):
        # Test with series containing NaN values
        import numpy as np

        open_with_nan = self.open.copy()
        open_with_nan.iloc[0:5] = np.nan

        result = pandas_ta.cpr(open_with_nan, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        # Should still have CPR columns
        self.assertIn("CPR_TC", result.columns)

    def test_cpr_option_offset(self):
        # Test option CPR with offset
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, offset=5
        )
        self.assertIsInstance(result, DataFrame)
        # First 5 rows should have NaN values due to offset
        self.assertTrue(result["OPT_TC_STRIKE"].iloc[0:5].isna().all())

    def test_cpr_option_fillna(self):
        # Test option CPR with fillna
        result = pandas_ta.cpr_option(
            self.open, self.high, self.low, self.close, fillna=0
        )
        self.assertIsInstance(result, DataFrame)
        # No NaN values in strike columns (filled with 0)
        self.assertFalse(result["OPT_TC_STRIKE"].isna().any())

    def test_decay(self):
        result = pandas_ta.decay(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LDECAY_5")

        result = pandas_ta.decay(self.close, mode="exp")
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EXPDECAY_5")

    def test_decreasing(self):
        result = pandas_ta.decreasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEC_1")

        result = pandas_ta.decreasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SDEC_3")

    def test_dpo(self):
        result = pandas_ta.dpo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DPO_20")

    def test_increasing(self):
        result = pandas_ta.increasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INC_1")

        result = pandas_ta.increasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINC_3")

    def test_long_run(self):
        result = pandas_ta.long_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_2")

    def test_psar(self):
        result = pandas_ta.psar(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PSAR_0.02_0.2")

        # Combine Long and Short SAR"s into one SAR value
        psar = result[result.columns[:2]].fillna(0)
        psar = psar[psar.columns[0]] + psar[psar.columns[1]]
        psar.name = result.name

        try:
            expected = tal.SAR(self.high, self.low)
            pdt.assert_series_equal(psar, expected)
        except AssertionError:
            try:
                psar_corr = pandas_ta.utils.df_error_analysis(
                    psar, expected, col=CORRELATION
                )
                self.assertGreater(psar_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(psar, CORRELATION, ex)

    def test_qstick(self):
        result = pandas_ta.qstick(self.open, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "QS_10")

    def test_short_run(self):
        result = pandas_ta.short_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SR_2")

    def test_ttm_trend(self):
        result = pandas_ta.ttm_trend(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TTMTREND_6")

    def test_vhf(self):
        result = pandas_ta.vhf(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VHF_28")

    def test_vortex(self):
        result = pandas_ta.vortex(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VTX_14")

    def test_pmax(self):
        result = pandas_ta.pmax(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PMAX_E_10_3.0")

    def test_tsignals(self):
        # Create a simple trend series for testing
        trend = pandas_ta.sma(self.close, length=10) - pandas_ta.sma(
            self.close, length=20
        )
        trend = (trend > 0).astype(int)
        result = pandas_ta.tsignals(trend)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TS")

    def test_xsignals(self):
        # Create simple signal series for testing
        signal = pandas_ta.rsi(self.close)
        result = pandas_ta.xsignals(signal, xa=70, xb=30)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "XS")
