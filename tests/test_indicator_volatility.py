from tests.assertions import assert_indicator_standard, assert_talib, IndicatorSpec
from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame, Series

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TestVolatility(TestCase):
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

    def test_aberration(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.aberration,
                args=[self.high, self.low, self.close],
                expected_name="ABER_5_15",
                expected_type=DataFrame,
                expected_columns=[
                    "ABER_ZG_5_15",
                    "ABER_SG_5_15",
                    "ABER_XG_5_15",
                    "ABER_ATR_5_15",
                ],
                none_arg_idx=None,
            ),
        )

    def test_cvi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cvi,
                args=[self.high, self.low],
                expected_name="CVI_10",
            ),
        )
        self.assertIsNone(pandas_ta.cvi(self.high, None))

    def test_hvol(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hvol,
                args=[self.close],
                expected_name="HVOL_20",
            ),
        )

    def test_accbands(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.accbands,
                args=[self.high, self.low, self.close],
                expected_name="ACCBANDS_20",
                expected_type=DataFrame,
                expected_columns=["ACCBL_20", "ACCBM_20", "ACCBU_20"],
                none_arg_idx=None,
            ),
        )

    def test_atr(self):
        result = pandas_ta.atr(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.ATR(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.atr,
                args=[self.high, self.low, self.close],
                expected_name="ATRr_14",
                none_arg_idx=None,
            ),
        )

    def test_bbands(self):
        result = pandas_ta.bbands(self.close, talib=False)
        if HAS_TALIB:
            bbu, bbm, bbl = talib.BBANDS(self.close)
            expecteddf = DataFrame({"BBL_5_2.0": bbl, "BBM_5_2.0": bbm, "BBU_5_2.0": bbu})
            assert_talib(
                self,
                result[["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0"]],
                expecteddf,
                correlation_threshold=0.99,
            )

        result = pandas_ta.bbands(self.close, ddof=0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        result = pandas_ta.bbands(self.close, ddof=1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.bbands,
                args=[self.close],
                expected_name="BBANDS_5_2.0",
                expected_type=DataFrame,
                expected_columns=[
                    "BBL_5_2.0",
                    "BBM_5_2.0",
                    "BBU_5_2.0",
                    "BBB_5_2.0",
                    "BBP_5_2.0",
                ],
            ),
        )

    def test_ce(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ce,
                args=[self.high, self.low, self.close],
                expected_name="CE_22_3.0",
                expected_type=DataFrame,
                expected_columns=["CE_L_22_3.0", "CE_S_22_3.0"],
                none_arg_idx=None,
            ),
        )

    def test_donchian(self):
        result = pandas_ta.donchian(self.high, self.low, lower_length=20, upper_length=5)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_5")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.donchian,
                args=[self.high, self.low],
                expected_name="DC_20_20",
                expected_type=DataFrame,
                expected_columns=["DCL_20_20", "DCM_20_20", "DCU_20_20"],
            ),
        )

    def test_kc(self):
        result = pandas_ta.kc(self.high, self.low, self.close, mamode="sma")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCs_20_2")

        result = pandas_ta.kc(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kc,
                args=[self.high, self.low, self.close],
                expected_name="KCe_20_2",
                expected_type=DataFrame,
                expected_columns=["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"],
            ),
        )

    def test_massi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.massi,
                args=[self.high, self.low],
                expected_name="MASSI_9_25",
                none_arg_idx=None,
            ),
        )

    def test_natr(self):
        result = pandas_ta.natr(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.NATR(self.high, self.low, self.close),
                # Native NATR uses EMA (default mamode='ema') while TA-Lib
                # NATR uses RMA; correlation is high but not ≥0.99.
                correlation_threshold=0.98,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.natr,
                args=[self.high, self.low, self.close],
                expected_name="NATR_14",
                none_arg_idx=None,
            ),
        )

    def test_pdist(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pdist,
                args=[self.open, self.high, self.low, self.close],
                expected_name="PDIST",
                none_arg_idx=None,
            ),
        )

    def test_rvi(self):
        result = pandas_ta.rvi(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIr_14")

        result = pandas_ta.rvi(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIt_14")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rvi,
                args=[self.close],
                expected_name="RVI_14",
                none_arg_idx=None,
            ),
        )

    def test_thermo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.thermo,
                args=[self.high, self.low],
                expected_name="THERMO_20_2_0.5",
                expected_type=DataFrame,
                expected_columns=[
                    "THERMO_20_2_0.5",
                    "THERMOma_20_2_0.5",
                    "THERMOl_20_2_0.5",
                    "THERMOs_20_2_0.5",
                ],
            ),
        )

    def test_true_range(self):
        result = pandas_ta.true_range(self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.TRANGE(self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.true_range,
                args=[self.high, self.low, self.close],
                expected_name="TRUERANGE_1",
                none_arg_idx=None,
            ),
        )

    def test_ui(self):
        result = pandas_ta.ui(self.close, everget=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UIe_14")

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ui,
                args=[self.close],
                expected_name="UI_14",
                none_arg_idx=None,
            ),
        )

    def test_hwc(self):
        result = pandas_ta.hwc(self.close, channel_eval=True)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("HWW", result.columns)
        self.assertIn("HWPCT", result.columns)

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.hwc,
                args=[self.close],
                expected_name="HWC",
                expected_type=DataFrame,
                expected_columns=["HWM", "HWU", "HWL"],
            ),
        )
