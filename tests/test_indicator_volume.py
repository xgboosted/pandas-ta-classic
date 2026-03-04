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


class TestVolume(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume_ = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume_
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ad(self):
        result = pandas_ta.ad(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")

        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")
        assert_offset(
            self,
            pandas_ta.ad,
            self.high,
            self.low,
            self.close,
            self.volume_,
            talib=False,
        )

    @talib_test
    def test_ad_talib(self):
        result = pandas_ta.ad(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        expected = tal.AD(self.high, self.low, self.close, self.volume_)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_ad_open(self):
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADo")

    def test_adosc(self):
        result = pandas_ta.adosc(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")

        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")
        assert_offset(
            self,
            pandas_ta.adosc,
            self.high,
            self.low,
            self.close,
            self.volume_,
            talib=False,
        )

    @talib_test
    def test_adosc_talib(self):
        result = pandas_ta.adosc(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        expected = tal.ADOSC(self.high, self.low, self.close, self.volume_)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_aobv(self):
        result = pandas_ta.aobv(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AOBVe_4_12_2_2_2")
        assert_columns(
            self,
            result,
            [
                "OBV",
                "OBV_min_2",
                "OBV_max_2",
                "OBVe_4",
                "OBVe_12",
                "AOBV_LR_2",
                "AOBV_SR_2",
            ],
        )
        assert_offset(self, pandas_ta.aobv, self.close, self.volume_)

        # slow < fast triggers swap (line 29: fast, slow = slow, fast)
        result_swap = pandas_ta.aobv(self.close, self.volume_, fast=12, slow=4)
        self.assertIsInstance(result_swap, DataFrame)

        # "length" kwarg is popped before passing to sub-indicators (line 36)
        result_len = pandas_ta.aobv(self.close, self.volume_, length=10)
        self.assertIsInstance(result_len, DataFrame)

    def test_cmf(self):
        result = pandas_ta.cmf(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CMF_20")
        assert_nan_count(self, result, 20)
        assert_offset(
            self, pandas_ta.cmf, self.high, self.low, self.close, self.volume_
        )

    def test_efi(self):
        result = pandas_ta.efi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EFI_13")
        assert_offset(self, pandas_ta.efi, self.close, self.volume_)

    def test_eom(self):
        result = pandas_ta.eom(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EOM_14_100000000")
        assert_nan_count(self, result, 14)
        assert_offset(
            self, pandas_ta.eom, self.high, self.low, self.close, self.volume_
        )

    def test_kvo(self):
        result = pandas_ta.kvo(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KVO_34_55_13")
        assert_offset(
            self, pandas_ta.kvo, self.high, self.low, self.close, self.volume_
        )
        assert_columns(self, result, ["KVO_34_55_13", "KVOs_34_55_13"])

    def test_mfi(self):
        result = pandas_ta.mfi(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")

        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")
        assert_nan_count(self, result, 14)
        assert_offset(
            self,
            pandas_ta.mfi,
            self.high,
            self.low,
            self.close,
            self.volume_,
            talib=False,
        )

    @talib_test
    def test_mfi_talib(self):
        result = pandas_ta.mfi(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        expected = tal.MFI(self.high, self.low, self.close, self.volume_)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_nvi(self):
        result = pandas_ta.nvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "NVI_1")
        assert_offset(self, pandas_ta.nvi, self.close, self.volume_)

    def test_obv(self):
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")

        result = pandas_ta.obv(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")
        assert_offset(self, pandas_ta.obv, self.close, self.volume_, talib=False)

    @talib_test
    def test_obv_talib(self):
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        expected = tal.OBV(self.close, self.volume_)
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
            self.assertGreater(corr, CORRELATION_THRESHOLD)

    def test_pvi(self):
        result = pandas_ta.pvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVI_1")
        assert_offset(self, pandas_ta.pvi, self.close, self.volume_)

    def test_pvol(self):
        result = pandas_ta.pvol(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVOL")
        assert_offset(self, pandas_ta.pvol, self.close, self.volume_)

    def test_pvr(self):
        result = pandas_ta.pvr(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVR")
        # sample indicator values from SPY
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 3)
        self.assertEqual(result.iloc[4], 2)
        self.assertEqual(result.iloc[6], 4)

    def test_pvt(self):
        result = pandas_ta.pvt(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVT")
        assert_offset(self, pandas_ta.pvt, self.close, self.volume_)

    def test_vp(self):
        result = pandas_ta.vp(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VP_10")

    def test_vp_sort_close(self):
        result = pandas_ta.vp(self.close, self.volume_, sort_close=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VP_10")

    def test_vfi(self):
        result = pandas_ta.vfi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VFI_130")
        assert_nan_count(self, result, 130)
        assert_offset(self, pandas_ta.vfi, self.close, self.volume_)
