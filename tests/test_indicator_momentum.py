from tests.assertions import assert_indicator_standard, assert_talib, IndicatorSpec
from tests.config import get_sample_data
import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame, Series

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TestMomentum(TestCase):
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

    def test_datetime_ordered(self):
        # Test if datetime64 index and ordered
        result = self.data.ta.datetime_ordered
        self.assertTrue(result)

        # Test if not ordered
        original = self.data.copy()
        reversal = original.ta.reverse
        result = reversal.ta.datetime_ordered
        self.assertFalse(result)

        # Test a non-datetime64 index
        original = self.data.copy()
        original.reset_index(inplace=True)
        result = original.ta.datetime_ordered
        self.assertFalse(result)

    def test_reverse(self):
        original = self.data.copy()
        result = original.ta.reverse

        # Check if first and last time are reversed
        self.assertEqual(result.index[-1], original.index[0])
        self.assertEqual(result.index[0], original.index[-1])

    def test_ao(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ao,
                args=[self.high, self.low],
                expected_name="AO_5_34",
                none_arg_idx=0,
            ),
        )

    def test_apo(self):
        result = pandas_ta.apo(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.APO(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.apo,
                args=[self.close],
                expected_name="APO_12_26",
                none_arg_idx=0,
            ),
        )

    def test_bias(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.bias,
                args=[self.close],
                expected_name="BIAS_SMA_26",
                none_arg_idx=0,
            ),
        )

    def test_bop(self):
        result = pandas_ta.bop(self.open, self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.BOP(self.open, self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.bop,
                args=[self.open, self.high, self.low, self.close],
                expected_name="BOP",
                none_arg_idx=0,
            ),
        )

    def test_brar(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.brar,
                args=[self.open, self.high, self.low, self.close],
                expected_name="BRAR_26",
                expected_type=DataFrame,
                expected_columns=["AR_26", "BR_26"],
                none_arg_idx=0,
            ),
        )

    def test_cci(self):
        result = pandas_ta.cci(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.CCI(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cci,
                args=[self.high, self.low, self.close],
                expected_name="CCI_14_0.015",
                none_arg_idx=0,
            ),
        )

    def test_cfo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cfo,
                args=[self.close],
                expected_name="CFO_9",
                none_arg_idx=0,
            ),
        )

    def test_cg(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cg,
                args=[self.close],
                expected_name="CG_10",
                none_arg_idx=0,
            ),
        )

    def test_cmo(self):
        # Native CMO uses rolling sum; TA-Lib CMO uses Wilder smoothing.
        # Correlation ~0.885 is expected between the two algorithms.
        result = pandas_ta.cmo(self.close)
        if HAS_TALIB:
            assert_talib(self, result, talib.CMO(self.close), correlation_threshold=0.85)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cmo,
                args=[self.close],
                expected_name="CMO_14",
                none_arg_idx=0,
            ),
        )

    def test_coppock(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.coppock,
                args=[self.close],
                expected_name="COPC_11_14_10",
                none_arg_idx=0,
            ),
        )

    def test_cti(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cti,
                args=[self.close],
                expected_name="CTI_12",
                none_arg_idx=0,
            ),
        )

    def test_er(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.er,
                args=[self.close],
                expected_name="ER_10",
                none_arg_idx=0,
            ),
        )

    def test_dm(self):
        result = pandas_ta.dm(self.high, self.low, talib=False)
        if HAS_TALIB:
            expecteddf = DataFrame(
                {
                    "DMP_14": talib.PLUS_DM(self.high, self.low),
                    "DMN_14": talib.MINUS_DM(self.high, self.low),
                }
            )
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dm,
                args=[self.high, self.low],
                expected_name="DM_14",
                expected_type=DataFrame,
                expected_columns=["DMP_14", "DMN_14"],
                none_arg_idx=0,
            ),
        )

    def test_eri(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.eri,
                args=[self.high, self.low, self.close],
                expected_name="ERI_13",
                expected_type=DataFrame,
                expected_columns=["BULLP_13", "BEARP_13"],
                none_arg_idx=0,
            ),
        )

    def test_fisher(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.fisher,
                args=[self.high, self.low],
                expected_name="FISHERT_9_1",
                expected_type=DataFrame,
                expected_columns=["FISHERT_9_1", "FISHERTs_9_1"],
                none_arg_idx=0,
            ),
        )

    def test_fosc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.fosc,
                args=[self.close],
                expected_name="FOSC_14",
                none_arg_idx=0,
            ),
        )

    def test_inertia(self):
        result = pandas_ta.inertia(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAr_20_14")

        result = pandas_ta.inertia(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAt_20_14")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.inertia,
                args=[self.close],
                expected_name="INERTIA_20_14",
                none_arg_idx=0,
            ),
        )

    def test_kdj(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kdj,
                args=[self.high, self.low, self.close],
                expected_name="KDJ_9_3",
                expected_type=DataFrame,
                expected_columns=["K_9_3", "D_9_3", "J_9_3"],
                none_arg_idx=0,
            ),
        )

    def test_kst(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kst,
                args=[self.close],
                expected_name="KST_10_15_20_30_10_10_10_15_9",
                expected_type=DataFrame,
                expected_columns=["KST_10_15_20_30_10_10_10_15", "KSTs_9"],
                none_arg_idx=0,
            ),
        )

    def test_macd(self):
        result = pandas_ta.macd(self.close, talib=False)
        if HAS_TALIB:
            macd_line, signal, hist = talib.MACD(self.close)
            expecteddf = DataFrame(
                {
                    "MACD_12_26_9": macd_line,
                    "MACDh_12_26_9": hist,
                    "MACDs_12_26_9": signal,
                }
            )
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.macd,
                args=[self.close],
                expected_name="MACD_12_26_9",
                expected_type=DataFrame,
                expected_columns=["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
                none_arg_idx=0,
            ),
        )

    def test_macdas(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.macd,
                args=[self.close],
                expected_name="MACDAS_12_26_9",
                expected_type=DataFrame,
                expected_columns=[
                    "MACDAS_12_26_9",
                    "MACDASh_12_26_9",
                    "MACDASs_12_26_9",
                ],
                none_arg_idx=0,
                kwargs={"asmode": True},
            ),
        )

    def test_macdext(self):
        # EMA-based should closely match regular MACD
        result = pandas_ta.macdext(self.close, talib=False)
        # SMA-based should give different values
        result_sma = pandas_ta.macdext(self.close, fastmatype=0, slowmatype=0, signalmatype=0, talib=False)
        self.assertIsInstance(result_sma, DataFrame)
        self.assertFalse(result["MACDEXT_12_26_9"].equals(result_sma["MACDEXT_12_26_9"]))
        if HAS_TALIB:
            result_tal = pandas_ta.macdext(self.close)
            self.assertIsInstance(result_tal, DataFrame)
            self.assertEqual(result_tal.name, "MACDEXT_12_26_9")
        pandas_ta.macdext(self.close, fillna=0)
        pandas_ta.macdext(self.close, fill_method="ffill")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.macdext,
                args=[self.close],
                expected_name="MACDEXT_12_26_9",
                expected_type=DataFrame,
                expected_columns=[
                    "MACDEXT_12_26_9",
                    "MACDEXTs_12_26_9",
                    "MACDEXTh_12_26_9",
                ],
                none_arg_idx=0,
            ),
        )

    def test_macdext_unsupported_matype_warns(self):
        import pytest

        with pytest.warns(UserWarning, match="EMA will be used instead"):
            result = pandas_ta.macdext(self.close, signalmatype=6, talib=False)
        self.assertIsInstance(result, DataFrame)

        with pytest.warns(UserWarning, match="EMA will be used instead"):
            result = pandas_ta.macdext(self.close, fastmatype=7, talib=False)
        self.assertIsInstance(result, DataFrame)

    def test_mom(self):
        result = pandas_ta.mom(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.MOM(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mom,
                args=[self.close],
                expected_name="MOM_10",
                none_arg_idx=0,
            ),
        )

    def test_pgo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pgo,
                args=[self.high, self.low, self.close],
                expected_name="PGO_14",
                none_arg_idx=0,
            ),
        )

    def test_ppo(self):
        result = pandas_ta.ppo(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result["PPO_12_26_9"],
                talib.PPO(self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ppo,
                args=[self.close],
                expected_name="PPO_12_26_9",
                expected_type=DataFrame,
                expected_columns=["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"],
                none_arg_idx=0,
            ),
        )

    def test_psl(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.psl,
                args=[self.close],
                expected_name="PSL_12",
                none_arg_idx=0,
            ),
        )

    def test_pvo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvo,
                args=[self.volume],
                expected_name="PVO_12_26_9",
                expected_type=DataFrame,
                expected_columns=["PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9"],
                none_arg_idx=None,
            ),
        )

    def test_qqe(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.qqe,
                args=[self.close],
                expected_name="QQE_14_5_4.236",
                expected_type=DataFrame,
                expected_columns=[
                    "QQE_14_5_4.236",
                    "QQE_14_5_4.236_RSIMA",
                    "QQEl_14_5_4.236",
                    "QQEs_14_5_4.236",
                    "QQEb_l_14_5_4.236",
                    "QQEb_s_14_5_4.236",
                    "QQEd_14_5_4.236",
                ],
                none_arg_idx=None,
            ),
        )

    def test_roc(self):
        result = pandas_ta.roc(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.ROC(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.roc,
                args=[self.close],
                expected_name="ROC_10",
                none_arg_idx=0,
            ),
        )

    def test_rocp(self):
        result = pandas_ta.rocp(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.ROCP(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rocp,
                args=[self.close],
                expected_name="ROCP_10",
                none_arg_idx=0,
            ),
        )

    def test_rocr(self):
        result = pandas_ta.rocr(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.ROCR(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rocr,
                args=[self.close],
                expected_name="ROCR_10",
                none_arg_idx=0,
            ),
        )

    def test_rocr100(self):
        result = pandas_ta.rocr100(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.ROCR100(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rocr100,
                args=[self.close],
                expected_name="ROCR100_10",
                none_arg_idx=0,
            ),
        )

    def test_rsi(self):
        result = pandas_ta.rsi(self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.RSI(self.close), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rsi,
                args=[self.close],
                expected_name="RSI_14",
                none_arg_idx=0,
            ),
        )

    def test_rsx(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rsx,
                args=[self.close],
                expected_name="RSX_14",
                none_arg_idx=None,
            ),
        )

    def test_rvgi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rvgi,
                args=[self.open, self.high, self.low, self.close],
                expected_name="RVGI_14_4",
                expected_type=DataFrame,
                expected_columns=["RVGIh_14_4", "RVGI_14_4", "RVGIs_14_4"],
                none_arg_idx=0,
            ),
        )

    def test_slope(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.slope,
                args=[self.close],
                expected_name="SLOPE_1",
                none_arg_idx=0,
            ),
        )

    def test_slope_as_angle(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.slope,
                args=[self.close],
                expected_name="ANGLEr_1",
                none_arg_idx=0,
                kwargs={"as_angle": True},
            ),
        )

    def test_slope_as_angle_to_degrees(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.slope,
                args=[self.close],
                expected_name="ANGLEd_1",
                none_arg_idx=0,
                kwargs={"as_angle": True, "to_degrees": True},
            ),
        )

    def test_smi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.smi,
                args=[self.close],
                expected_name="SMI_5_20_5",
                expected_type=DataFrame,
                expected_columns=["SMI_5_20_5", "SMIs_5_20_5", "SMIo_5_20_5"],
                none_arg_idx=0,
            ),
        )

    def test_smi_scalar(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.smi,
                args=[self.close],
                expected_name="SMI_5_20_5_10.0",
                expected_type=DataFrame,
                expected_columns=[
                    "SMI_5_20_5_10.0",
                    "SMIs_5_20_5_10.0",
                    "SMIo_5_20_5_10.0",
                ],
                none_arg_idx=0,
                kwargs={"scalar": 10},
            ),
        )

    def test_squeeze(self):
        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5")

        result = pandas_ta.squeeze(self.high, self.low, self.close, lazybear=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5_LB")

        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False, lazybear=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5_LB")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.squeeze,
                args=[self.high, self.low, self.close],
                expected_name="SQZ_20_2.0_20_1.5",
                expected_type=DataFrame,
                expected_columns=[
                    "SQZ_20_2.0_20_1.5",
                    "SQZ_ON",
                    "SQZ_OFF",
                    "SQZ_NO",
                ],
                none_arg_idx=0,
            ),
        )

    def test_squeeze_pro(self):
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_2_1.5_1")

        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPRO_20_2.0_20_3.0_2.0_1.0")

        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_3.0_2.0_1.0")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.squeeze_pro,
                args=[self.high, self.low, self.close],
                expected_name="SQZPRO_20_2.0_20_2_1.5_1",
                expected_type=DataFrame,
                expected_columns=[
                    "SQZPRO_20_2.0_20_2_1.5_1",
                    "SQZPRO_ON_WIDE",
                    "SQZPRO_ON_NORMAL",
                    "SQZPRO_ON_NARROW",
                    "SQZPRO_OFF",
                    "SQZPRO_NO",
                ],
                none_arg_idx=0,
            ),
        )

    def test_stc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stc,
                args=[self.close],
                expected_name="STC_10_12_26_0.5",
                expected_type=DataFrame,
                expected_columns=[
                    "STC_10_12_26_0.5",
                    "STCmacd_10_12_26_0.5",
                    "STCstoch_10_12_26_0.5",
                ],
                none_arg_idx=0,
            ),
        )

    def test_stoch(self):
        # TV Correlation
        result = pandas_ta.stoch(self.high, self.low, self.close)
        if HAS_TALIB:
            stochk, stochd = talib.STOCH(self.high, self.low, self.close, 14, 3, 0, 3, 0)
            expecteddf = DataFrame({"STOCHk_14_3_0_3_0": stochk, "STOCHd_14_3_0_3": stochd})
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stoch,
                args=[self.high, self.low, self.close],
                expected_name="STOCH_14_3_3",
                expected_type=DataFrame,
                expected_columns=["STOCHk_14_3_3", "STOCHd_14_3_3"],
                none_arg_idx=0,
            ),
        )

    def test_stochf(self):
        result = pandas_ta.stochf(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            stochfk, stochfd = talib.STOCHF(self.high, self.low, self.close, 5, 3, 0)
            expecteddf = DataFrame({"STOCHFk_5_3": stochfk, "STOCHFd_5_3": stochfd})
            assert_talib(self, result, expecteddf, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stochf,
                args=[self.high, self.low, self.close],
                expected_name="STOCHF_5_3",
                expected_type=DataFrame,
                expected_columns=["STOCHFk_5_3", "STOCHFd_5_3"],
                none_arg_idx=0,
            ),
        )

    def test_stochrsi(self):
        # TV Correlation
        result = pandas_ta.stochrsi(self.close)
        if HAS_TALIB:
            _stochrsi_k, stochrsi_d = talib.STOCHRSI(self.close, 14, 14, 3, 0)
            assert_talib(self, result.iloc[:, 0], stochrsi_d, correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.stochrsi,
                args=[self.close],
                expected_name="STOCHRSI_14_14_3_3",
                expected_type=DataFrame,
                expected_columns=["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"],
                none_arg_idx=0,
            ),
        )

    def test_td_seq(self):
        """TS Sequential: Working but SLOW implementation"""
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.td_seq,
                args=[self.close],
                expected_name="TD_SEQ",
                expected_type=DataFrame,
                expected_columns=["TD_SEQ_UPa", "TD_SEQ_DNa"],
                none_arg_idx=None,
            ),
        )

    def test_trix(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.trix,
                args=[self.close],
                expected_name="TRIX_30_9",
                expected_type=DataFrame,
                expected_columns=["TRIX_30_9", "TRIXs_30_9"],
                none_arg_idx=0,
            ),
        )

    def test_tsi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tsi,
                args=[self.close],
                expected_name="TSI_13_25_13",
                expected_type=DataFrame,
                expected_columns=["TSI_13_25_13", "TSIs_13_25_13"],
                none_arg_idx=0,
            ),
        )

    def test_uo(self):
        result = pandas_ta.uo(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.ULTOSC(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.uo,
                args=[self.high, self.low, self.close],
                expected_name="UO_7_14_28",
                none_arg_idx=0,
            ),
        )

    def test_willr(self):
        result = pandas_ta.willr(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.WILLR(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.willr,
                args=[self.high, self.low, self.close],
                expected_name="WILLR_14",
                none_arg_idx=0,
            ),
        )

    def test_lrsi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.lrsi,
                args=[self.close],
                expected_name="LRSI_14",
                none_arg_idx=None,
            ),
        )

    def test_po(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.po,
                args=[self.close],
                expected_name="PO_14",
                none_arg_idx=0,
            ),
        )

    def test_trixh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.trixh,
                args=[self.close],
                expected_name="TRIXH_18_9",
                expected_type=DataFrame,
                expected_columns=["TRIX_18_9", "TRIXs_18_9", "TRIXh_18_9"],
                none_arg_idx=0,
            ),
        )

    def test_vwmacd(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vwmacd,
                args=[self.close, self.volume],
                expected_name="VWMACD_12_26_9",
                expected_type=DataFrame,
                expected_columns=[
                    "VWMACD_12_26_9",
                    "VWMACDh_12_26_9",
                    "VWMACDs_12_26_9",
                ],
                none_arg_idx=0,
            ),
        )
