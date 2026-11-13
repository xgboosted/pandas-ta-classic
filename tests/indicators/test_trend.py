from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.utils._cpr import get_previous_period_ohlcv
from tests.assertions import IndicatorSpec, assert_all_nan, assert_indicator_standard, assert_talib
from tests.config import get_sample_data
from tests.contracts import spy as _spy

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


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
        if HAS_TALIB:
            assert_talib(self, result.iloc[:, 0], talib.ADX(self.high, self.low, self.close))
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.adx,
                args=[self.high, self.low, self.close],
                expected_name="ADX_14",
                expected_type=DataFrame,
                expected_columns=["ADX_14", "DMP_14", "DMN_14"],
            ),
        )

    def test_adxr(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.adxr,
                args=[self.high, self.low, self.close],
                expected_name="ADXR_14",
                expected_type=DataFrame,
                expected_columns=["ADXR_14", "DMP_14", "DMN_14"],
            ),
        )

    def test_amat(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.amat,
                args=[self.close],
                expected_name="AMATe_8_21_2",
                expected_type=DataFrame,
                expected_columns=["AMATe_LR_8_21_2", "AMATe_SR_8_21_2"],
                none_arg_idx=None,
            ),
        )

    def test_aroon(self):
        result = pandas_ta.aroon(self.high, self.low, talib=False)
        if HAS_TALIB:
            aroond, aroonu = talib.AROON(self.high, self.low)
            expecteddf = DataFrame({"AROOND_14": aroond, "AROONU_14": aroonu})
            assert_talib(self, result.iloc[:, :2], expecteddf)  # TA-Lib's AROON has no oscillator column
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.aroon,
                args=[self.high, self.low],
                expected_name="AROON_14",
                expected_type=DataFrame,
                expected_columns=["AROOND_14", "AROONU_14", "AROONOSC_14"],
            ),
        )

    def test_aroon_osc(self):
        result = pandas_ta.aroon(self.high, self.low)
        if HAS_TALIB:
            assert_talib(self, result.iloc[:, 2], talib.AROONOSC(self.high, self.low))

    def test_chop(self):
        result = pandas_ta.chop(self.high, self.low, self.close, ln=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOPln_14_1_100")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.chop,
                args=[self.high, self.low, self.close],
                expected_name="CHOP_14_1_100",
                kwargs={"ln": False},
            ),
        )

    def test_cksp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cksp,
                args=[self.high, self.low, self.close],
                expected_name="CKSP_10_3_20",
                expected_type=DataFrame,
                expected_columns=["CKSPl_10_3_20", "CKSPs_10_3_20"],
                kwargs={"tvmode": False},
            ),
        )

    def test_cksp_tv(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cksp,
                args=[self.high, self.low, self.close],
                expected_name="CKSP_10_1_9",
                expected_type=DataFrame,
                expected_columns=["CKSPl_10_1_9", "CKSPs_10_1_9"],
                kwargs={"tvmode": True},
            ),
        )

    def test_cpr_basic(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"levels": "basic"},
            ),
        )

    def test_cpr_classic_standard(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"method": "classic", "levels": "standard"},
            ),
        )

    def test_cpr_classic_extended(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_R3",
                    "CPR_R4",
                    "CPR_S3",
                    "CPR_S4",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"method": "classic", "levels": "extended"},
            ),
        )

    def test_cpr_camarilla(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_R3",
                    "CPR_R4",
                    "CPR_S3",
                    "CPR_S4",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"method": "camarilla", "levels": "all"},
            ),
        )

    def test_cpr_fibonacci(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_R3",
                    "CPR_S3",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"method": "fibonacci", "levels": "all"},
            ),
        )

    def test_cpr_woodie(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"method": "woodie", "levels": "standard"},
            ),
        )

    def test_cpr_width_analysis(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"width_analysis": True},
            ),
        )

    def test_cpr_price_position(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"price_position": True},
            ),
        )

    def test_cpr_virgin_detection(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                    "CPR_VIRGIN",
                ],
                kwargs={"virgin_cpr": True, "virgin_lookforward": 5},
            ),
        )
        virgin_values = result["CPR_VIRGIN"].dropna()
        if len(virgin_values) > 0:
            self.assertTrue(virgin_values.dtype == bool)

    def test_cpr_virgin_disabled(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                ],
                kwargs={"virgin_cpr": False},
            ),
        )

    def test_cpr_virgin_custom_lookforward(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cpr,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CPR",
                expected_type=DataFrame,
                expected_columns=[
                    "CPR_TC",
                    "CPR_PIVOT",
                    "CPR_BC",
                    "CPR_R1",
                    "CPR_R2",
                    "CPR_S1",
                    "CPR_S2",
                    "CPR_WIDTH",
                    "CPR_WIDTH_PCT",
                    "CPR_WIDTH_CLASS",
                    "CPR_POSITION",
                    "CPR_VIRGIN",
                ],
                kwargs={"virgin_cpr": True, "virgin_lookforward": 10},
            ),
        )

    def test_cpr_invalid_method(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) method must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, method="invalid_method")

    def test_cpr_invalid_timeframe(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) timeframe must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, timeframe="invalid_timeframe")

    def test_cpr_invalid_levels(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) levels must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, levels="invalid_levels")

    def test_cpr_empty_series(self):
        empty_series = Series(dtype=float)
        with self.assertLogs("pandas_ta_classic.utils._core", level="WARNING") as cm:
            result = pandas_ta.cpr(empty_series, empty_series, empty_series, empty_series)
        self.assertGreaterEqual(len(cm.output), 1)
        self.assertTrue(
            any("Series has 0 rows" in message for message in cm.output),
            f"Expected empty-series validation warning in logs: {cm.output}",
        )
        assert_all_nan(self, result)

    def test_cpr_with_nans(self):
        open_with_nan = self.open.copy()
        open_with_nan.iloc[0:5] = np.nan
        result = pandas_ta.cpr(open_with_nan, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_TC", result.columns)

    def test_decay(self):
        result = pandas_ta.decay(self.close, mode="exp")
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EXPDECAY_5")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.decay,
                args=[self.close],
                expected_name="LDECAY_5",
            ),
        )

    def test_edecay(self):
        result = pandas_ta.edecay(self.close, length=10)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EDECAY_10")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.edecay,
                args=[self.close],
                expected_name="EDECAY_5",
            ),
        )

    def test_decreasing(self):
        result = pandas_ta.decreasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SDEC_3")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.decreasing,
                args=[self.close],
                expected_name="DEC_1",
            ),
        )

    def test_dpo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dpo,
                args=[self.close],
                expected_name="DPO_20",
                none_arg_idx=None,
            ),
        )

    def test_increasing(self):
        result = pandas_ta.increasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINC_3")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.increasing,
                args=[self.close],
                expected_name="INC_1",
            ),
        )

    def test_long_run(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.long_run,
                args=[self.close, self.open],
                expected_name="LR_2",
                none_arg_idx=None,
            ),
        )
        with self.assertRaisesRegex(ValueError, "long_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.long_run(None, self.open)
        with self.assertRaisesRegex(ValueError, "long_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.long_run(self.close, None)

    def test_dx(self):
        result = pandas_ta.dx(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.DX(self.high, self.low, self.close))
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dx,
                args=[self.high, self.low, self.close],
                expected_name="DX_14",
            ),
        )

    def test_minus_dm(self):
        result = pandas_ta.minus_dm(self.high, self.low, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.MINUS_DM(self.high, self.low))
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minus_dm,
                args=[self.high, self.low],
                expected_name="MINUS_DM_14",
            ),
        )

    def test_plus_dm(self):
        result = pandas_ta.plus_dm(self.high, self.low, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.PLUS_DM(self.high, self.low))
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.plus_dm,
                args=[self.high, self.low],
                expected_name="PLUS_DM_14",
            ),
        )

    def test_psar_rejects_talib(self):
        """TA-Lib's SAR returns the stop alone: psar(talib=True) built PSARaf as NaN on
        every bar and PSARr as 0.0, which hid 521 reversals on SPY_D. The native stop
        equals SAR bar for bar, so psar has no talib parameter, and passing one raises."""
        for value in (True, False):
            with self.assertRaisesRegex(TypeError, "talib"):
                pandas_ta.psar(self.high, self.low, self.close, talib=value)

    def test_psar(self):
        result = pandas_ta.psar(self.high, self.low)
        if HAS_TALIB:
            # TA-Lib's SAR is one line; native splits it into long and short. combine_first,
            # unlike the old fillna(0) sum, keeps the first bar NaN where both are.
            psar_combined = result.iloc[:, 0].combine_first(result.iloc[:, 1])
            assert_talib(self, psar_combined, talib.SAR(self.high, self.low))
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.psar,
                args=[self.high, self.low],
                expected_name="PSAR_0.02_0.2",
                expected_type=DataFrame,
                expected_columns=[
                    "PSARl_0.02_0.2",
                    "PSARs_0.02_0.2",
                    "PSARaf_0.02_0.2",
                    "PSARr_0.02_0.2",
                ],
                none_arg_idx=None,
            ),
        )

    def test_qstick(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.qstick,
                args=[self.open, self.close],
                expected_name="QS_10",
            ),
        )

    def test_sarext(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sarext,
                args=[self.high, self.low],
                expected_name="SAREXT",
            ),
        )

    def test_short_run(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.short_run,
                args=[self.close, self.open],
                expected_name="SR_2",
                none_arg_idx=None,
            ),
        )
        with self.assertRaisesRegex(ValueError, "short_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.short_run(None, self.open)
        with self.assertRaisesRegex(ValueError, "short_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.short_run(self.close, None)

    def test_ttm_trend(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ttm_trend,
                args=[self.high, self.low, self.close],
                expected_name="TTMTREND_6",
                expected_type=DataFrame,
                expected_columns=["TTM_TRND_6"],
            ),
        )

    def test_ttm_trend_missing_close_is_nan(self):
        # `close > avg` is False for a NaN close, which used to read as a -1
        # downtrend on the missing bar between two uptrend bars.
        high = Series(np.arange(20) + 2.0)
        low = Series(np.arange(20) * 1.0)
        close = Series(np.arange(20) + 1.0)
        close[12] = np.nan
        result = pandas_ta.ttm_trend(high, low, close)["TTM_TRND_6"]
        assert result.iloc[6:12].eq(1).all() and result.iloc[13:].eq(1).all()
        assert np.isnan(result.iloc[12])

    def test_vhf(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vhf,
                args=[self.close],
                expected_name="VHF_28",
            ),
        )

    def test_vortex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vortex,
                args=[self.high, self.low, self.close],
                expected_name="VTX_14",
                expected_type=DataFrame,
                expected_columns=["VTXP_14", "VTXM_14"],
            ),
        )

    def test_pmax(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pmax,
                args=[self.high, self.low, self.close],
                expected_name="PMAX_E_10_3.0",
            ),
        )

    def test_tsignals(self):
        trend = pandas_ta.sma(self.close, length=10) - pandas_ta.sma(self.close, length=20)
        trend = (trend > 0).astype(int)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tsignals,
                args=[trend],
                expected_name="TS",
                expected_type=DataFrame,
                expected_columns=["TS_Trends", "TS_Trades", "TS_Entries", "TS_Exits"],
                none_arg_idx=None,
            ),
        )
        with self.assertRaisesRegex(ValueError, "tsignals\\(\\) requires a 'trend' Series"):
            pandas_ta.tsignals(None)

    def test_xsignals(self):
        signal = pandas_ta.rsi(self.close)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.xsignals,
                args=[signal],
                expected_name="XS",
                expected_type=DataFrame,
                expected_columns=["TS_Trends", "TS_Trades", "TS_Entries", "TS_Exits"],
                none_arg_idx=None,
                kwargs={"xa": 70, "xb": 30},
            ),
        )
        with self.assertRaisesRegex(ValueError, "xsignals\\(\\) requires a 'signal' Series"):
            pandas_ta.xsignals(None, 70, 30)

    def test_tsignals_gap_in_the_trend_makes_no_trade(self):
        # The gap read as "no trend": an exit at bar 3 and a re-entry at bar 4.
        result = pandas_ta.tsignals(Series([0, 1, 1, np.nan, 1, 0.0]))
        self.assertEqual(result["TS_Trends"].tolist(), [0, 1, 1, 1, 1, 0])
        self.assertEqual(result["TS_Trades"].tolist(), [0, 1, 0, 0, 0, -1])

    def test_xsignals_crossing_on_a_gap_is_kept(self):
        # The up-cross through 20 fell on the missing bar, so the whole trade
        # (entry at bar 2, exit at bar 5) was lost and bars 3-4 read no trend.
        signal = Series([10, 15, np.nan, 30, 25, 15, 10.0])
        result = pandas_ta.xsignals(signal, 20, 20)
        self.assertEqual(result["TS_Trends"].tolist(), [0, 0, 0, 1, 1, 0, 0])
        self.assertEqual(result["TS_Entries"].tolist(), [0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(result["TS_Exits"].tolist(), [0, 0, 0, 0, 0, 1, 0])


needs_talib = pytest.mark.skipif(not pandas_ta.Imports["talib"], reason="TA-Lib not installed")


# ---------------------------------------------------------------------------
# edecay is multiplicative decay floored at close (not additive)
# ---------------------------------------------------------------------------


class TestEdecayFormula(TestCase):
    """edecay uses multiplicative exp decay: result[i] = max(close[i], prev * exp(-1/n))."""

    def test_edecay_decays_after_spike(self):
        """After a spike, edecay must decay exponentially, not stay flat."""
        length = 5
        # Flat series then a spike: value should decay after the spike
        close = pd.Series([100.0] * 10 + [200.0] + [100.0] * 10)
        result = pandas_ta.edecay(close, length=length)
        self.assertIsNotNone(result)

        # Indices 11-13: decay path still > 100 floor (floor takes over at ~14)
        post_spike = result.iloc[11:14].values
        self.assertTrue(
            (post_spike > 100.0).all(),
            "edecay must stay above floor (close) right after spike",
        )
        self.assertTrue(
            (post_spike < 200.0).all(),
            "edecay must decay below spike peak",
        )

    def test_edecay_decay_rate_matches_formula(self):
        """Decay rate must match exp(-1/length) when close is below decay path."""
        import math

        length = 5
        factor = math.exp(-1.0 / length)
        # Spike then flat low floor — ensures decay path dominates
        close_vals = [10.0] * 5 + [1000.0] + [10.0] * 20
        close = pd.Series(close_vals)
        result = pandas_ta.edecay(close, length=length)

        # Two consecutive decaying bars: ratio must equal factor
        i = 6  # first bar after spike where decay path > close
        j = 7
        # Assert preconditions explicitly so a broken edecay can't silently pass
        self.assertGreater(
            result.iloc[i],
            10.0,
            f"Precondition failed: result[{i}]={result.iloc[i]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        self.assertGreater(
            result.iloc[j],
            10.0,
            f"Precondition failed: result[{j}]={result.iloc[j]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        ratio = result.iloc[j] / result.iloc[i]
        self.assertAlmostEqual(
            ratio,
            factor,
            places=6,
            msg=f"edecay ratio {ratio:.6f} must equal exp(-1/{length})={factor:.6f}",
        )

    def test_edecay_never_below_close(self):
        """edecay result must always be >= close (floored at close)."""
        close = pd.Series([50.0 + i % 20 for i in range(50)], dtype=float)
        result = pandas_ta.edecay(close, length=5)
        mask = result.notna()
        self.assertTrue(
            (result[mask].values >= close[mask].values - 1e-9).all(),
            "edecay must never fall below close (floor at close)",
        )


# ---------------------------------------------------------------------------
# psar — invalid optional close returns None (not crash)
# ---------------------------------------------------------------------------


class TestPsarCloseNoneGuard(TestCase):
    """psar() optional close None guard: invalid close → None, not AttributeError."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]

    def test_invalid_close_list_raises(self):
        """A plain list as close is rejected by verify_series (issue #145)."""
        bad_close = list(self.high.values)
        with self.assertRaisesRegex(TypeError, r"psar\(\) expected a pandas Series but got list"):
            pandas_ta.psar(high=self.high, low=self.low, close=bad_close)

    def test_valid_call_without_close_returns_dataframe(self):
        """Normal call without close returns a DataFrame."""
        result = pandas_ta.psar(high=self.high, low=self.low)
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# cpr previous-period levels
# ---------------------------------------------------------------------------


def _ramp(index: pd.DatetimeIndex) -> pd.DataFrame:
    c = np.arange(len(index), dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=index)


def _previous_period_reference(df: pd.DataFrame, keys: list) -> np.ndarray:
    """Loop reference: first/max/min/last/sum of the previous observed period."""
    order = list(dict.fromkeys(keys))
    rows = {k: df[[x == k for x in keys]] for k in order}
    out = []
    for k in keys:
        j = order.index(k)
        if j == 0:
            out.append([np.nan] * 5)
            continue
        p = rows[order[j - 1]]
        out.append([p.open.iloc[0], p.high.max(), p.low.min(), p.close.iloc[-1], p.volume.sum()])
    return np.array(out)


_BDAYS = pd.bdate_range("2024-01-01", "2024-06-28")


_SESSIONS = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * k) for d in pd.bdate_range("2024-03-04", periods=12) for k in range(78)])


_WEEK = lambda t: t.isocalendar()[:2]


_MONTH = lambda t: (t.year, t.month)


_DAY = lambda t: t.date()


@pytest.mark.parametrize(
    "index, timeframe, key",
    [
        pytest.param(_BDAYS, "weekly", _WEEK, id="weekly-weekday-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-06-30", freq="D"), "weekly", _WEEK, id="weekly-7-day-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"), "weekly", _WEEK, id="weekly-hourly-24-7"),
        pytest.param(_BDAYS.drop(pd.bdate_range("2024-03-25", "2024-03-29")), "weekly", _WEEK, id="weekly-missing-week"),
        pytest.param(_BDAYS, "monthly", _MONTH, id="monthly-weekday-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"), "monthly", _MONTH, id="monthly-hourly-24-7"),
        pytest.param(_SESSIONS, "intraday", _DAY, id="intraday-incl-mondays"),
        pytest.param(_SESSIONS.tz_localize("America/New_York"), "intraday", _DAY, id="intraday-tz-aware"),
    ],
)
def test_cpr_previous_period_is_last_completed_period(index, timeframe, key):
    df = _ramp(index)
    got = get_previous_period_ohlcv(df, timeframe)[["prev_open", "prev_high", "prev_low", "prev_close", "prev_volume"]].to_numpy(float)
    expected = _previous_period_reference(df, [key(t) for t in df.index])
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_cpr_weekly_pivot_on_spy():
    d = _spy().iloc[-400:]
    week = d.index.to_period("W-SUN")
    prev = d.groupby(week).agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).shift(1).reindex(week).set_axis(d.index)
    expected = (prev.high + prev.low + prev.close) / 3
    got = pandas_ta.cpr(d.open, d.high, d.low, d.close, timeframe="weekly")["CPR_PIVOT"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


# ---------------------------------------------------------------------------
# sarext native vs TA-Lib, every parameter
# ---------------------------------------------------------------------------


_SAREXT_CASES = {
    "defaults": {},
    "offsetonreverse": {"offsetonreverse": 0.01},
    "startvalue-long": {"startvalue": 100.0},
    "startvalue-short": {"startvalue": -150.0},
    "long-acceleration": {"accelerationinitlong": 0.01, "accelerationlong": 0.03, "accelerationmaxlong": 0.25},
    "short-acceleration": {"accelerationinitshort": 0.04, "accelerationshort": 0.01, "accelerationmaxshort": 0.1},
    "init-above-max": {"accelerationinitlong": 0.5, "accelerationmaxlong": 0.2, "accelerationshort": 0.5, "accelerationmaxshort": 0.3},
}


def _random_walk_hl(seed: int, n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # From 200, not 100: seed 0's walk fell to -11.85, and sarext reads NaN, with a
    # warning, where high or low is at or below zero (#345). A shift changes no SAR step.
    c = 200 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"high": np.maximum(o, c) + rng.random(n), "low": np.minimum(o, c) - rng.random(n)})


@needs_talib
@pytest.mark.parametrize("kwargs", list(_SAREXT_CASES.values()), ids=list(_SAREXT_CASES))
@pytest.mark.parametrize("data", ["spy", "walk0", "walk3"])
def test_sarext_native_matches_talib(kwargs, data):
    import talib

    d = _spy() if data == "spy" else _random_walk_hl(int(data[-1]))
    got = pandas_ta.sarext(d.high, d.low, talib=False, **kwargs).to_numpy(float)
    expected = talib.SAREXT(d.high.to_numpy(float), d.low.to_numpy(float), **kwargs)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10, equal_nan=True)


# ---------------------------------------------------------------------------
# cpr interval removal
# ---------------------------------------------------------------------------


def test_cpr_interval_is_rejected():
    d = _spy().iloc[-200:]
    with pytest.raises(TypeError, match="interval"):
        pandas_ta.cpr(d.open, d.high, d.low, d.close, timeframe="weekly", interval="1d")
    with pytest.raises(TypeError, match="interval"):
        d.copy().ta.cpr(timeframe="weekly", interval="1d")
