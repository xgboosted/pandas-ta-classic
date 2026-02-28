from tests.config import (
    assert_columns,
    assert_nan_count,
    assert_offset,
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
        result = pandas_ta.ao(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AO_5_34")
        assert_offset(self, pandas_ta.ao, self.high, self.low)

    def test_apo(self):
        result = pandas_ta.apo(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "APO_12_26")

        try:
            expected = tal.APO(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.apo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "APO_12_26")
        assert_offset(self, pandas_ta.apo, self.close, talib=False)

    def test_bias(self):
        result = pandas_ta.bias(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "BIAS_SMA_26")
        assert_offset(self, pandas_ta.bias, self.close)

    def test_bop(self):
        result = pandas_ta.bop(self.open, self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "BOP")

        try:
            expected = tal.BOP(self.open, self.high, self.low, self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.bop(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "BOP")
        assert_offset(
            self, pandas_ta.bop, self.open, self.high, self.low, self.close, talib=False
        )

    def test_brar(self):
        result = pandas_ta.brar(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BRAR_26")
        assert_offset(
            self,
            pandas_ta.brar,
            self.open,
            self.high,
            self.low,
            self.close,
            expected_cols=["AR_26", "BR_26"],
        )

    def test_cci(self):
        result = pandas_ta.cci(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CCI_14_0.015")

        try:
            expected = tal.CCI(self.high, self.low, self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.cci(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CCI_14_0.015")
        assert_offset(self, pandas_ta.cci, self.high, self.low, self.close, talib=False)

    def test_cfo(self):
        result = pandas_ta.cfo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CFO_9")
        assert_offset(self, pandas_ta.cfo, self.close)

    def test_cg(self):
        result = pandas_ta.cg(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CG_10")
        assert_offset(self, pandas_ta.cg, self.close)

    def test_cmo(self):
        result = pandas_ta.cmo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CMO_14")

        try:
            expected = tal.CMO(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.cmo(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CMO_14")
        assert_offset(self, pandas_ta.cmo, self.close, talib=False)

    def test_coppock(self):
        result = pandas_ta.coppock(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "COPC_11_14_10")
        assert_offset(self, pandas_ta.coppock, self.close)

    def test_cti(self):
        result = pandas_ta.cti(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CTI_12")
        assert_offset(self, pandas_ta.cti, self.close)

    def test_er(self):
        result = pandas_ta.er(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ER_10")
        assert_offset(self, pandas_ta.er, self.close)

    def test_dm(self):
        result = pandas_ta.dm(self.high, self.low, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DM_14")

        try:
            expected_pos = tal.PLUS_DM(self.high, self.low)
            expected_neg = tal.MINUS_DM(self.high, self.low)
            expecteddf = DataFrame({"DMP_14": expected_pos, "DMN_14": expected_neg})
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                dmp = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION
                )
                self.assertGreater(dmp, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

            try:
                dmn = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION
                )
                self.assertGreater(dmn, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.dm(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DM_14")
        assert_offset(self, pandas_ta.dm, self.high, self.low, talib=False)
        assert_columns(
            self, pandas_ta.dm(self.high, self.low, talib=False), ["DMP_14", "DMN_14"]
        )

    def test_eri(self):
        result = pandas_ta.eri(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ERI_13")
        assert_offset(
            self,
            pandas_ta.eri,
            self.high,
            self.low,
            self.close,
            expected_cols=["BULLP_13", "BEARP_13"],
        )

    def test_fisher(self):
        result = pandas_ta.fisher(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "FISHERT_9_1")
        assert_offset(
            self,
            pandas_ta.fisher,
            self.high,
            self.low,
            expected_cols=["FISHERT_9_1", "FISHERTs_9_1"],
        )

    def test_inertia(self):
        result = pandas_ta.inertia(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIA_20_14")

        result = pandas_ta.inertia(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAr_20_14")

        result = pandas_ta.inertia(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INERTIAt_20_14")

    def test_kdj(self):
        result = pandas_ta.kdj(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KDJ_9_3")
        assert_offset(
            self,
            pandas_ta.kdj,
            self.high,
            self.low,
            self.close,
            expected_cols=["K_9_3", "D_9_3", "J_9_3"],
        )

    def test_kst(self):
        result = pandas_ta.kst(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KST_10_15_20_30_10_10_10_15_9")
        assert_offset(
            self,
            pandas_ta.kst,
            self.close,
            expected_cols=["KST_10_15_20_30_10_10_10_15", "KSTs_9"],
        )

    def test_macd(self):
        result = pandas_ta.macd(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MACD_12_26_9")

        try:
            expected = tal.MACD(self.close)
            expecteddf = DataFrame(
                {
                    "MACD_12_26_9": expected[0],
                    "MACDh_12_26_9": expected[2],
                    "MACDs_12_26_9": expected[1],
                }
            )
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                macd_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION
                )
                self.assertGreater(macd_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                history_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION
                )
                self.assertGreater(history_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

            try:
                signal_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 2], expecteddf.iloc[:, 2], col=CORRELATION
                )
                self.assertGreater(signal_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 2], CORRELATION, ex, newline=False)

        result = pandas_ta.macd(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MACD_12_26_9")
        assert_offset(self, pandas_ta.macd, self.close, talib=False)
        assert_columns(
            self,
            pandas_ta.macd(self.close, talib=False),
            ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        )

    def test_macdas(self):
        result = pandas_ta.macd(self.close, asmode=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "MACDAS_12_26_9")

    def test_mom(self):
        result = pandas_ta.mom(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MOM_10")

        try:
            expected = tal.MOM(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.mom(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MOM_10")
        assert_offset(self, pandas_ta.mom, self.close, talib=False)

    def test_pgo(self):
        result = pandas_ta.pgo(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PGO_14")

    def test_ppo(self):
        result = pandas_ta.ppo(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PPO_12_26_9")

        try:
            expected = tal.PPO(self.close)
            pdt.assert_series_equal(result["PPO_12_26_9"], expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result["PPO_12_26_9"], expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result["PPO_12_26_9"], CORRELATION, ex)

        result = pandas_ta.ppo(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PPO_12_26_9")
        assert_columns(self, result, ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"])
        assert_offset(self, pandas_ta.ppo, self.close, talib=False)

        # fast > slow triggers swap (line 29: fast, slow = slow, fast)
        result_swap = pandas_ta.ppo(self.close, fast=26, slow=12, talib=False)
        self.assertIsInstance(result_swap, DataFrame)
        self.assertEqual(result_swap.name, "PPO_12_26_9")

    def test_psl(self):
        result = pandas_ta.psl(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PSL_12")

    def test_pvo(self):
        result = pandas_ta.pvo(self.volume)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PVO_12_26_9")
        assert_offset(
            self,
            pandas_ta.pvo,
            self.volume,
            expected_cols=["PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9"],
        )

    def test_qqe(self):
        result = pandas_ta.qqe(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "QQE_14_5_4.236")
        # QQE is stateful (iterative loop) so skip the offset-shift comparison;
        # exercise fill branches and None-guard manually instead
        self.assertIsNone(pandas_ta.qqe(None))
        pandas_ta.qqe(self.close, fillna=0)
        pandas_ta.qqe(self.close, fill_method="ffill")
        pandas_ta.qqe(self.close, fill_method="bfill")

    def test_roc(self):
        result = pandas_ta.roc(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ROC_10")

        try:
            expected = tal.ROC(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.roc(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ROC_10")
        assert_offset(self, pandas_ta.roc, self.close, talib=False)

    def test_rsi(self):
        result = pandas_ta.rsi(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RSI_14")

        try:
            expected = tal.RSI(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.rsi(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RSI_14")
        assert_offset(self, pandas_ta.rsi, self.close, talib=False)

    def test_rsx(self):
        result = pandas_ta.rsx(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RSX_14")
        assert_offset(self, pandas_ta.rsx, self.close)

    def test_rvgi(self):
        result = pandas_ta.rvgi(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "RVGI_14_4")
        assert_offset(
            self,
            pandas_ta.rvgi,
            self.open,
            self.high,
            self.low,
            self.close,
            expected_cols=["RVGIh_14_4", "RVGI_14_4", "RVGIs_14_4"],
        )

    def test_slope(self):
        result = pandas_ta.slope(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SLOPE_1")

    def test_slope_as_angle(self):
        result = pandas_ta.slope(self.close, as_angle=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ANGLEr_1")

    def test_slope_as_angle_to_degrees(self):
        result = pandas_ta.slope(self.close, as_angle=True, to_degrees=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ANGLEd_1")

    def test_smi(self):
        result = pandas_ta.smi(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SMI_5_20_5")
        self.assertEqual(len(result.columns), 3)

    def test_smi_scalar(self):
        result = pandas_ta.smi(self.close, scalar=10)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SMI_5_20_5_10.0")
        self.assertEqual(len(result.columns), 3)

    def test_squeeze(self):
        result = pandas_ta.squeeze(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5")
        assert_columns(self, result, ["SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO"])
        assert_offset(
            self,
            pandas_ta.squeeze,
            self.high,
            self.low,
            self.close,
        )

        # asint=False keeps bool flags as bool — bypasses the astype(int) block
        result_bool = pandas_ta.squeeze(self.high, self.low, self.close, asint=False)
        self.assertIsInstance(result_bool, DataFrame)

        # mamode="ema" exercises the ema branch instead of sma (line 80)
        result = pandas_ta.squeeze(self.high, self.low, self.close, mamode="ema")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5")

        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5")

        result = pandas_ta.squeeze(self.high, self.low, self.close, lazybear=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZ_20_2.0_20_1.5_LB")

        result = pandas_ta.squeeze(
            self.high, self.low, self.close, tr=False, lazybear=True
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZhlr_20_2.0_20_1.5_LB")

        # detailed=True covers the extended series branches (lines 163-254)
        result = pandas_ta.squeeze(self.high, self.low, self.close, detailed=True)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("SQZ_INC", result.columns)
        self.assertIn("SQZ_DEC", result.columns)
        self.assertIn("SQZ_PINC", result.columns)
        self.assertIn("SQZ_PDEC", result.columns)
        self.assertIn("SQZ_NDEC", result.columns)
        self.assertIn("SQZ_NINC", result.columns)

        # fill branches inside the detailed block (lines 186-247)
        pandas_ta.squeeze(self.high, self.low, self.close, detailed=True, fillna=0)
        pandas_ta.squeeze(
            self.high, self.low, self.close, detailed=True, fill_method="ffill"
        )
        pandas_ta.squeeze(
            self.high, self.low, self.close, detailed=True, fill_method="bfill"
        )

    def test_squeeze_pro(self):
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPRO_20_2.0_20_2_1.5_1")
        assert_columns(
            self,
            result,
            ["SQZPRO_20_2.0_20_2_1.5_1", "SQZPRO_ON_WIDE", "SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO"],
        )
        assert_offset(
            self,
            pandas_ta.squeeze_pro,
            self.high,
            self.low,
            self.close,
        )

        # invalid kc_scalar ordering triggers the valid_kc_scaler guard (line 61-62)
        self.assertIsNone(
            pandas_ta.squeeze_pro(
                self.high, self.low, self.close,
                kc_scalar_wide=1, kc_scalar_normal=1.5, kc_scalar_narrow=2,
            )
        )

        # asint=False keeps bool flags as bool — bypasses the astype(int) block
        result_bool = pandas_ta.squeeze_pro(
            self.high, self.low, self.close, asint=False
        )
        self.assertIsInstance(result_bool, DataFrame)

        # mamode="ema" exercises the ema branch (line 113)
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, mamode="ema")
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_2_1.5_1")

        result = pandas_ta.squeeze_pro(
            self.high, self.low, self.close, 20, 2, 20, 3, 2, 1
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPRO_20_2.0_20_3.0_2.0_1.0")

        result = pandas_ta.squeeze_pro(
            self.high, self.low, self.close, 20, 2, 20, 3, 2, 1, tr=False
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SQZPROhlr_20_2.0_20_3.0_2.0_1.0")

        # detailed=True covers the extended series branches (lines 223-314)
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, detailed=True)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("SQZPRO_INC", result.columns)
        self.assertIn("SQZPRO_DEC", result.columns)
        self.assertIn("SQZPRO_PINC", result.columns)
        self.assertIn("SQZPRO_PDEC", result.columns)
        self.assertIn("SQZPRO_NDEC", result.columns)
        self.assertIn("SQZPRO_NINC", result.columns)

        # fill branches inside the detailed block (lines 246-307)
        pandas_ta.squeeze_pro(
            self.high, self.low, self.close, detailed=True, fillna=0
        )
        pandas_ta.squeeze_pro(
            self.high, self.low, self.close, detailed=True, fill_method="ffill"
        )
        pandas_ta.squeeze_pro(
            self.high, self.low, self.close, detailed=True, fill_method="bfill"
        )

    def test_stc(self):
        result = pandas_ta.stc(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "STC_10_12_26_0.5")
        assert_offset(
            self,
            pandas_ta.stc,
            self.close,
            expected_cols=[
                "STC_10_12_26_0.5",
                "STCmacd_10_12_26_0.5",
                "STCstoch_10_12_26_0.5",
            ],
        )

    def test_stoch(self):
        # TV Correlation
        result = pandas_ta.stoch(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "STOCH_14_3_3")
        assert_offset(self, pandas_ta.stoch, self.high, self.low, self.close)
        assert_columns(self, result, ["STOCHk_14_3_3", "STOCHd_14_3_3"])

        try:
            expected = tal.STOCH(self.high, self.low, self.close, 14, 3, 0, 3, 0)
            expecteddf = DataFrame(
                {"STOCHk_14_3_0_3_0": expected[0], "STOCHd_14_3_0_3": expected[1]}
            )
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                stochk_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION
                )
                self.assertGreater(stochk_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                stochd_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION
                )
                self.assertGreater(stochd_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

    def test_stochrsi(self):
        # TV Correlation
        result = pandas_ta.stochrsi(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "STOCHRSI_14_14_3_3")
        assert_columns(self, result, ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"])
        assert_offset(self, pandas_ta.stochrsi, self.close)

        # mamode="ema" exercises the ema smoothing branch
        result_ema = pandas_ta.stochrsi(self.close, mamode="ema")
        self.assertIsInstance(result_ema, DataFrame)

        try:
            expected = tal.STOCHRSI(self.close, 14, 14, 3, 0)
            expecteddf = DataFrame(
                {"STOCHRSIk_14_14_0_3": expected[0], "STOCHRSId_14_14_3_0": expected[1]}
            )
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                stochrsid_corr = pandas_ta.utils.df_error_analysis(
                    result.iloc[:, 0], expecteddf.iloc[:, 1], col=CORRELATION
                )
                self.assertGreater(stochrsid_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex, newline=False)

    def test_td_seq(self):
        """TS Sequential: Working but SLOW implementation"""
        result = pandas_ta.td_seq(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TD_SEQ")

    def test_trix(self):
        result = pandas_ta.trix(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TRIX_30_9")

        result_14 = pandas_ta.trix(self.close, length=14)
        self.assertIsInstance(result_14, DataFrame)
        trix_col = result_14["TRIX_14_9"]
        try:
            expected = tal.TRIX(self.close, timeperiod=14)
            pdt.assert_series_equal(trix_col, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    trix_col, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(trix_col, CORRELATION, ex)

        assert_offset(self, pandas_ta.trix, self.close)
        assert_nan_count(self, trix_col, 3 * 14 - 2)

    def test_tsi(self):
        result = pandas_ta.tsi(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TSI_13_25_13")
        assert_offset(
            self,
            pandas_ta.tsi,
            self.close,
            expected_cols=["TSI_13_25_13", "TSIs_13_25_13"],
        )

    def test_uo(self):
        result = pandas_ta.uo(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UO_7_14_28")

        try:
            expected = tal.ULTOSC(self.high, self.low, self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.uo(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UO_7_14_28")

    def test_willr(self):
        result = pandas_ta.willr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WILLR_14")

        try:
            expected = tal.WILLR(self.high, self.low, self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(
                    result, expected, col=CORRELATION
                )
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

    def test_lrsi(self):
        result = pandas_ta.lrsi(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRSI_14")
        assert_offset(self, pandas_ta.lrsi, self.close)

    def test_po(self):
        result = pandas_ta.po(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PO_14")

    def test_trixh(self):
        result = pandas_ta.trixh(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TRIXH_18_9")

    def test_vwmacd(self):
        result = pandas_ta.vwmacd(self.close, self.volume)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VWMACD_12_26_9")

        result = pandas_ta.willr(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WILLR_14")
