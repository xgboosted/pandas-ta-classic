from tests.assertions import assert_indicator_standard, assert_talib, IndicatorSpec
from tests.config import get_sample_data
import pandas_ta_classic as pandas_ta

import warnings
from unittest import TestCase
from pandas import DataFrame, Series

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


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
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.alma,
                args=[self.close],
                expected_name="ALMA_10_6.0_0.85",
            ),
        )

    def test_dema(self):
        result = pandas_ta.dema(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.DEMA(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dema,
                args=[self.close],
                expected_name="DEMA_10",
            ),
        )

    def test_ema(self):
        if HAS_TALIB:
            expected = talib.EMA(self.close, 10)
            result_presma = pandas_ta.ema(self.close, presma=False)
            assert_talib(self, result_presma, expected, correlation_threshold=0.99)
            result_talib_false = pandas_ta.ema(self.close, talib=False)
            assert_talib(self, result_talib_false, expected, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ema,
                args=[self.close],
                expected_name="EMA_10",
            ),
        )

    def test_fwma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.fwma,
                args=[self.close],
                expected_name="FWMA_10",
            ),
        )

    def test_hilo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hilo,
                args=[self.high, self.low, self.close],
                expected_name="HILO_13_21",
                expected_type=DataFrame,
                expected_columns=["HILO_13_21", "HILOl_13_21", "HILOs_13_21"],
            ),
        )

    def test_hl2(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hl2,
                args=[self.high, self.low],
                expected_name="HL2",
                none_arg_idx=None,
            ),
        )

    def test_hlc3(self):
        result = pandas_ta.hlc3(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.TYPPRICE(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hlc3,
                args=[self.high, self.low, self.close],
                expected_name="HLC3",
                none_arg_idx=None,
            ),
        )

    def test_hma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hma,
                args=[self.close],
                expected_name="HMA_10",
            ),
        )

    def test_ht_trendline(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_trendline,
                args=[self.close],
                expected_name="HT_TRENDLINE",
                none_arg_idx=0,
            ),
        )

    def test_hwma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hwma,
                args=[self.close],
                expected_name="HWMA_0.2_0.1_0.1",
                none_arg_idx=None,
            ),
        )

    def test_kama(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kama,
                args=[self.close],
                expected_name="KAMA_10_2_30",
            ),
        )

    def test_jma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.jma,
                args=[self.close],
                expected_name="JMA_7_0",
            ),
        )

    def test_ichimoku(self):
        # Legacy tuple return (as_dataframe defaults to None) still works but
        # emits a DeprecationWarning.
        with self.assertWarns(DeprecationWarning):
            ichimoku, span = pandas_ta.ichimoku(self.high, self.low, self.close)
        self.assertIsInstance(ichimoku, DataFrame)
        self.assertIsInstance(span, DataFrame)
        self.assertEqual(ichimoku.name, "ICHIMOKU_9_26_52")
        self.assertEqual(span.name, "ICHISPAN_9_26")

    def test_ichimoku_as_dataframe(self):
        # Default append_span=False: visible period only, no future-dated rows.
        result = pandas_ta.ichimoku(self.high, self.low, self.close, as_dataframe=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ICHIMOKU_9_26_52")
        self.assertEqual(result.category, "overlap")
        self.assertEqual(
            list(result.columns),
            ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
        )
        self.assertEqual(len(result), len(self.close))

    def test_ichimoku_as_dataframe_append_span(self):
        kijun = 26
        result = pandas_ta.ichimoku(self.high, self.low, self.close, as_dataframe=True, append_span=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ICHIMOKU_9_26_52")
        self.assertEqual(
            list(result.columns),
            ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
        )
        # Length = visible rows (len close) + future-dated span rows (kijun).
        self.assertEqual(len(result), len(self.close) + kijun)
        # Span rows: only ISA/ISB populated; ITS/IKS/ICS are NaN.
        span_rows = result.iloc[-kijun:]
        self.assertTrue(span_rows[["ITS_9", "IKS_26", "ICS_26"]].isna().all().all())
        self.assertTrue(span_rows["ISA_9"].notna().any())

    def test_ichimoku_as_dataframe_false_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = pandas_ta.ichimoku(self.high, self.low, self.close, as_dataframe=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_linreg(self):
        result = pandas_ta.linreg(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.LINEARREG(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.linreg,
                args=[self.close],
                expected_name="LR_14",
            ),
        )

    def test_linreg_angle(self):
        result = pandas_ta.linreg(self.close, angle=True, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.LINEARREG_ANGLE(self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.linreg,
                args=[self.close],
                expected_name="LRa_14",
                kwargs={"angle": True},
                none_arg_idx=None,
            ),
        )

    def test_linreg_intercept(self):
        result = pandas_ta.linreg(self.close, intercept=True, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.LINEARREG_INTERCEPT(self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.linreg,
                args=[self.close],
                expected_name="LRb_14",
                kwargs={"intercept": True},
                none_arg_idx=None,
            ),
        )

    def test_linreg_r(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.linreg,
                args=[self.close],
                expected_name="LRr_14",
                kwargs={"r": True},
                none_arg_idx=None,
            ),
        )

    def test_linreg_slope(self):
        result = pandas_ta.linreg(self.close, slope=True, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.LINEARREG_SLOPE(self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.linreg,
                args=[self.close],
                expected_name="LRm_14",
                kwargs={"slope": True},
                none_arg_idx=None,
            ),
        )

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

    def test_mama(self):
        result = pandas_ta.mama(self.close)
        if HAS_TALIB:
            mama, fama = talib.MAMA(self.close)
            expecteddf = DataFrame({"MAMA_0.5_0.05": mama, "FAMA_0.5_0.05": fama})
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mama,
                args=[self.close],
                expected_name="MAMA_0.5_0.05",
                expected_type=DataFrame,
                expected_columns=["MAMA_0.5_0.05", "FAMA_0.5_0.05"],
            ),
        )

    def test_mavp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mavp,
                args=[self.close],
                expected_name="MAVP_2_30",
            ),
        )

    def test_mavp_unsupported_mamode_warns(self):
        import pytest

        with pytest.warns(UserWarning, match="Results will use SMA"):
            result = pandas_ta.mavp(self.close, mamode=1, talib=False)
        self.assertIsInstance(result, Series)

    def test_mcgd(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mcgd,
                args=[self.close],
                expected_name="MCGD_10",
            ),
        )

    def test_midpoint(self):
        result = pandas_ta.midpoint(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.MIDPOINT(self.close, 2), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.midpoint,
                args=[self.close],
                expected_name="MIDPOINT_2",
            ),
        )

    def test_midprice(self):
        result = pandas_ta.midprice(self.high, self.low, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.MIDPRICE(self.high, self.low, 2),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.midprice,
                args=[self.high, self.low],
                expected_name="MIDPRICE_2",
            ),
        )

    def test_ohlc4(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ohlc4,
                args=[self.open, self.high, self.low, self.close],
                expected_name="OHLC4",
                none_arg_idx=None,
            ),
        )

    def test_pwma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pwma,
                args=[self.close],
                expected_name="PWMA_10",
            ),
        )

    def test_rma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rma,
                args=[self.close],
                expected_name="RMA_10",
            ),
        )

    def test_sinwma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sinwma,
                args=[self.close],
                expected_name="SINWMA_14",
            ),
        )

    def test_sma(self):
        result = pandas_ta.sma(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.SMA(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sma,
                args=[self.close],
                expected_name="SMA_10",
            ),
        )

    def test_ssf(self):
        result = pandas_ta.ssf(self.close, poles=2)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_2")
        result = pandas_ta.ssf(self.close, poles=3)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_3")
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ssf,
                args=[self.close],
                expected_name="SSF_10_2",
                kwargs={"poles": 2},
            ),
        )

    def test_swma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.swma,
                args=[self.close],
                expected_name="SWMA_10",
            ),
        )

    def test_supertrend(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.supertrend,
                args=[self.high, self.low, self.close],
                expected_name="SUPERT_7_3.0",
                expected_type=DataFrame,
                expected_columns=[
                    "SUPERT_7_3.0",
                    "SUPERTd_7_3.0",
                    "SUPERTl_7_3.0",
                    "SUPERTs_7_3.0",
                ],
            ),
        )

    def test_t3(self):
        result = pandas_ta.t3(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.T3(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.t3,
                args=[self.close],
                expected_name="T3_10_0.7",
            ),
        )

    def test_tema(self):
        result = pandas_ta.tema(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.TEMA(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tema,
                args=[self.close],
                expected_name="TEMA_10",
            ),
        )

    def test_trima(self):
        result = pandas_ta.trima(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.TRIMA(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.trima,
                args=[self.close],
                expected_name="TRIMA_10",
            ),
        )

    def test_tsf(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tsf,
                args=[self.close],
                expected_name="TSF_14",
                none_arg_idx=0,
            ),
        )

    def test_vidya(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vidya,
                args=[self.close],
                expected_name="VIDYA_14",
            ),
        )

    def test_vwap(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vwap,
                args=[self.high, self.low, self.close, self.volume],
                expected_name="VWAP_D",
                none_arg_idx=None,
            ),
        )

    def test_vwma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vwma,
                args=[self.close, self.volume],
                expected_name="VWMA_10",
            ),
        )

    def test_wcp(self):
        result = pandas_ta.wcp(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.WCLPRICE(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.wcp,
                args=[self.high, self.low, self.close],
                expected_name="WCP",
                none_arg_idx=None,
            ),
        )

    def test_wma(self):
        result = pandas_ta.wma(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.WMA(self.close, 10), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.wma,
                args=[self.close],
                expected_name="WMA_10",
            ),
        )

    def test_zlma(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.zlma,
                args=[self.close],
                expected_name="ZL_EMA_10",
            ),
        )

    def test_mmar(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mmar,
                args=[self.close],
                expected_name="MMAR_10_5_6",
                expected_type=DataFrame,
                expected_columns=[
                    "MMAR_10",
                    "MMAR_15",
                    "MMAR_20",
                    "MMAR_25",
                    "MMAR_30",
                    "MMAR_35",
                ],
                none_arg_idx=None,
            ),
        )

    def test_rainbow(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rainbow,
                args=[self.close],
                expected_name="RAINBOW_2_10",
                expected_type=DataFrame,
                expected_columns=[
                    "RAINBOW_1",
                    "RAINBOW_2",
                    "RAINBOW_3",
                    "RAINBOW_4",
                    "RAINBOW_5",
                    "RAINBOW_6",
                    "RAINBOW_7",
                    "RAINBOW_8",
                    "RAINBOW_9",
                    "RAINBOW_10",
                ],
                none_arg_idx=None,
            ),
        )
