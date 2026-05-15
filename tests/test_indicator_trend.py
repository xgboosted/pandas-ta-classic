from tests.assertions import assert_indicator_standard, assert_talib, IndicatorSpec
from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase, skip
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
        if HAS_TALIB:
            assert_talib(
                self,
                result.iloc[:, 0],
                tal.ADX(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
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
            aroond, aroonu = tal.AROON(self.high, self.low)
            expecteddf = DataFrame({"AROOND_14": aroond, "AROONU_14": aroonu})
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
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
            assert_talib(
                self,
                result.iloc[:, 2],
                tal.AROONOSC(self.high, self.low),
                correlation_threshold=0.99,
            )

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
                kwargs={"method": "invalid_method"},
            ),
        )

    def test_cpr_invalid_timeframe(self):
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
                kwargs={"timeframe": "invalid_timeframe"},
            ),
        )

    def test_cpr_invalid_levels(self):
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
                kwargs={"levels": "invalid_levels"},
            ),
        )

    def test_cpr_empty_series(self):
        from pandas import Series

        empty_series = Series(dtype=float)
        with self.assertLogs("pandas_ta_classic.utils._core", level="WARNING") as cm:
            result = pandas_ta.cpr(
                empty_series, empty_series, empty_series, empty_series
            )
        self.assertGreaterEqual(len(cm.output), 1)
        self.assertTrue(
            any("Series has 0 rows" in message for message in cm.output),
            f"Expected empty-series validation warning in logs: {cm.output}",
        )
        self.assertIsNone(result)

    def test_cpr_with_nans(self):
        import numpy as np

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
            ),
        )

    def test_dx(self):
        result = pandas_ta.dx(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                tal.DX(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
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
            assert_talib(
                self,
                result,
                tal.MINUS_DM(self.high, self.low),
                correlation_threshold=0.99,
            )
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
            assert_talib(
                self,
                result,
                tal.PLUS_DM(self.high, self.low),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.plus_dm,
                args=[self.high, self.low],
                expected_name="PLUS_DM_14",
            ),
        )

    def test_psar(self):
        result = pandas_ta.psar(self.high, self.low)
        if HAS_TALIB:
            psar_combined = result[result.columns[:2]].fillna(0)
            psar_combined = (
                psar_combined[psar_combined.columns[0]]
                + psar_combined[psar_combined.columns[1]]
            )
            psar_combined.name = result.name
            assert_talib(
                self,
                psar_combined,
                tal.SAR(self.high, self.low),
                correlation_threshold=0.99,
            )
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
            ),
        )

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
        trend = pandas_ta.sma(self.close, length=10) - pandas_ta.sma(
            self.close, length=20
        )
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
