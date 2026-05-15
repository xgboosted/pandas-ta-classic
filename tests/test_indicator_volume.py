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
        if HAS_TALIB:
            assert_talib(self, result, tal.AD(self.high, self.low, self.close, self.volume_), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ad,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="AD",
                none_arg_idx=None,
            ),
        )

    def test_ad_open(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ad,
                args=[self.high, self.low, self.close, self.volume_, self.open],
                expected_name="ADo",
                none_arg_idx=None,
            ),
        )

    def test_adosc(self):
        result = pandas_ta.adosc(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        if HAS_TALIB:
            assert_talib(self, result, tal.ADOSC(self.high, self.low, self.close, self.volume_), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.adosc,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="ADOSC_3_10",
            ),
        )

    def test_aobv(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.aobv,
                args=[self.close, self.volume_],
                expected_name="AOBVe_4_12_2_2_2",
                expected_type=DataFrame,
                expected_columns=[
                    "OBV",
                    "OBV_min_2",
                    "OBV_max_2",
                    "OBVe_4",
                    "OBVe_12",
                    "AOBV_LR_2",
                    "AOBV_SR_2",
                ],
                none_arg_idx=None,
            ),
        )

    def test_cmf(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cmf,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="CMF_20",
            ),
        )

    def test_efi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.efi,
                args=[self.close, self.volume_],
                expected_name="EFI_13",
            ),
        )

    def test_emv(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.emv,
                args=[self.high, self.low, self.volume_],
                expected_name="EMV",
            ),
        )

    def test_eom(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.eom,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="EOM_14_100000000",
            ),
        )

    def test_kvo(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.kvo,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="KVO_34_55_13",
                expected_type=DataFrame,
                expected_columns=["KVO_34_55_13", "KVOs_34_55_13"],
            ),
        )

    def test_marketfi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.marketfi,
                args=[self.high, self.low, self.volume_],
                expected_name="MARKETFI",
            ),
        )
        self.assertIsNone(pandas_ta.marketfi(self.high, None, self.volume_))
        self.assertIsNone(pandas_ta.marketfi(self.high, self.low, None))

    def test_mfi(self):
        result = pandas_ta.mfi(
            self.high, self.low, self.close, self.volume_, talib=False
        )
        if HAS_TALIB:
            assert_talib(self, result, tal.MFI(self.high, self.low, self.close, self.volume_), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mfi,
                args=[self.high, self.low, self.close, self.volume_],
                expected_name="MFI_14",
            ),
        )

    def test_nvi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.nvi,
                args=[self.close, self.volume_],
                expected_name="NVI_1",
            ),
        )

    def test_obv(self):
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, tal.OBV(self.close, self.volume_), correlation_threshold=0.99)
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.obv,
                args=[self.close, self.volume_],
                expected_name="OBV",
                none_arg_idx=None,
            ),
        )

    def test_pvi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvi,
                args=[self.close, self.volume_],
                expected_name="PVI_1",
            ),
        )

    def test_pvol(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvol,
                args=[self.close, self.volume_],
                expected_name="PVOL",
                none_arg_idx=None,
            ),
        )

    def test_pvr(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvr,
                args=[self.close, self.volume_],
                expected_name="PVR",
                none_arg_idx=None,
            ),
        )
        # sample indicator values from SPY
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 3)
        self.assertEqual(result.iloc[4], 2)
        self.assertEqual(result.iloc[6], 4)

    def test_pvt(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.pvt,
                args=[self.close, self.volume_],
                expected_name="PVT",
                none_arg_idx=None,
            ),
        )

    def test_vp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vp,
                args=[self.close, self.volume_],
                expected_name="VP_10",
                expected_type=DataFrame,
                expected_columns=[
                    "low_close",
                    "mean_close",
                    "high_close",
                    "pos_volume",
                    "neg_volume",
                    "total_volume",
                ],
                none_arg_idx=None,
            ),
        )

    def test_vfi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vfi,
                args=[self.close, self.volume_],
                expected_name="VFI_130",
            ),
        )

    def test_vosc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vosc,
                args=[self.volume_],
                expected_name="VOSC_14_28",
            ),
        )

    def test_wad(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.wad,
                args=[self.high, self.low, self.close],
                expected_name="WAD",
            ),
        )
        self.assertIsNone(pandas_ta.wad(self.high, None, self.close))
        self.assertIsNone(pandas_ta.wad(self.high, self.low, None))
