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
        result = pandas_ta.aberration(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ABER_5_15")
        assert_offset(self, pandas_ta.aberration, self.high, self.low, self.close)
        assert_columns(
            self,
            result,
            ["ABER_ZG_5_15", "ABER_SG_5_15", "ABER_XG_5_15", "ABER_ATR_5_15"],
        )

    def test_accbands(self):
        result = pandas_ta.accbands(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ACCBANDS_20")
        assert_offset(self, pandas_ta.accbands, self.high, self.low, self.close)
        assert_columns(self, result, ["ACCBL_20", "ACCBM_20", "ACCBU_20"])

    def test_atr(self):
        result = pandas_ta.atr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ATRr_14")

        result = pandas_ta.atr(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ATRr_14")
        assert_offset(self, pandas_ta.atr, self.high, self.low, self.close, talib=False)
        atr_result = pandas_ta.atr(self.high, self.low, self.close, talib=False)
        assert_nan_count(self, atr_result, 14)

    @talib_test
    def test_atr_talib(self):
        result = pandas_ta.atr(self.high, self.low, self.close, talib=False)
        expected = tal.ATR(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_bbands(self):
        result = pandas_ta.bbands(self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        result = pandas_ta.bbands(self.close, ddof=0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        result = pandas_ta.bbands(self.close, ddof=1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")
        assert_offset(self, pandas_ta.bbands, self.close, talib=False)
        bbands_result = pandas_ta.bbands(self.close, talib=False)
        assert_columns(
            self,
            bbands_result,
            ["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "BBB_5_2.0", "BBP_5_2.0"],
        )

    @talib_test
    def test_bbands_talib(self):
        result = pandas_ta.bbands(self.close, talib=False)
        expected = tal.BBANDS(self.close)
        for col, expected_series in zip(
            ["BBU_5_2.0", "BBM_5_2.0", "BBL_5_2.0"], expected
        ):
            corr = pandas_ta.utils.df_error_analysis(
                result[col], expected_series, col=CORRELATION
            )
            self.assertGreater(corr, CORRELATION_THRESHOLD)
        self.assertTrue(result["BBB_5_2.0"].dropna().gt(0).all())
        self.assertTrue(result["BBP_5_2.0"].dropna().between(0, 1).all())

    def test_donchian(self):
        result = pandas_ta.donchian(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_20")

        result = pandas_ta.donchian(
            self.high, self.low, lower_length=20, upper_length=5
        )
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_5")
        assert_offset(self, pandas_ta.donchian, self.high, self.low)
        donchian_result = pandas_ta.donchian(self.high, self.low)
        assert_columns(self, donchian_result, ["DCL_20_20", "DCM_20_20", "DCU_20_20"])

    def test_kc(self):
        result = pandas_ta.kc(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCe_20_2")

        result = pandas_ta.kc(self.high, self.low, self.close, mamode="sma")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCs_20_2")
        assert_offset(self, pandas_ta.kc, self.high, self.low, self.close)
        kc_result = pandas_ta.kc(self.high, self.low, self.close)
        assert_columns(self, kc_result, ["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"])

    def test_massi(self):
        result = pandas_ta.massi(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MASSI_9_25")
        assert_offset(self, pandas_ta.massi, self.high, self.low)

    def test_natr(self):
        result = pandas_ta.natr(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "NATR_14")

        result = pandas_ta.natr(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "NATR_14")
        assert_offset(
            self, pandas_ta.natr, self.high, self.low, self.close, talib=False
        )

    @talib_test
    def test_natr_talib(self):
        result = pandas_ta.natr(self.high, self.low, self.close, talib=False)
        expected = tal.NATR(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_pdist(self):
        result = pandas_ta.pdist(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PDIST")
        assert_offset(self, pandas_ta.pdist, self.open, self.high, self.low, self.close)

    def test_rvi(self):
        result = pandas_ta.rvi(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVI_14")

        result = pandas_ta.rvi(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIr_14")

        result = pandas_ta.rvi(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIt_14")
        assert_offset(self, pandas_ta.rvi, self.close)

    def test_thermo(self):
        result = pandas_ta.thermo(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "THERMO_20_2_0.5")
        assert_offset(self, pandas_ta.thermo, self.high, self.low)
        assert_columns(
            self,
            result,
            [
                "THERMO_20_2_0.5",
                "THERMOma_20_2_0.5",
                "THERMOl_20_2_0.5",
                "THERMOs_20_2_0.5",
            ],
        )

    def test_true_range(self):
        result = pandas_ta.true_range(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRUERANGE_1")

        result = pandas_ta.true_range(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRUERANGE_1")
        assert_offset(
            self, pandas_ta.true_range, self.high, self.low, self.close, talib=False
        )

    @talib_test
    def test_true_range_talib(self):
        result = pandas_ta.true_range(self.high, self.low, self.close, talib=False)
        expected = tal.TRANGE(self.high, self.low, self.close)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ui(self):
        result = pandas_ta.ui(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UI_14")

        result = pandas_ta.ui(self.close, everget=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UIe_14")
        assert_offset(self, pandas_ta.ui, self.close)

    def test_hwc(self):
        result = pandas_ta.hwc(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HWC")
        assert_offset(self, pandas_ta.hwc, self.close)
        assert_columns(self, result, ["HWM", "HWU", "HWL"])

    def test_hwc_channel_eval(self):
        result = pandas_ta.hwc(self.close, channel_eval=True)
        self.assertIsInstance(result, DataFrame)
        assert_columns(self, result, ["HWM", "HWU", "HWL", "HWW", "HWPCT"])
