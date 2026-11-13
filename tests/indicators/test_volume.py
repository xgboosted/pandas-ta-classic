from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from tests.assertions import IndicatorSpec, assert_indicator_standard, assert_talib
from tests.config import get_sample_data

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


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
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.AD(self.high, self.low, self.close, self.volume_))
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
        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.ADOSC(self.high, self.low, self.close, self.volume_))
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
        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_, talib=False)
        if HAS_TALIB:
            # TA-Lib 0.8 (not 0.7.1) is off by up to 6.7 on 28 bars between 333 and 458,
            # where native matches a plain loop over the definition to 3e-14; only those
            # bars are left out, so the warm-up is still compared.
            keep = np.r_[0:333, 459 : len(result)]
            assert_talib(
                self,
                result.iloc[keep],
                talib.MFI(self.high, self.low, self.close, self.volume_).iloc[keep],
            )
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
            assert_talib(self, result, talib.OBV(self.close, self.volume_))
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
        # Bar 0 has no difference to rank; it read 1, a rising close, before.
        self.assertTrue(result.isna().iloc[0])
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

    def test_vp_sort_close(self):
        chronological = pandas_ta.vp(self.close, self.volume_)
        by_price = pandas_ta.vp(self.close, self.volume_, sort_close=True)
        self.assertIsInstance(by_price, DataFrame)
        self.assertEqual(list(by_price.columns), list(chronological.columns))
        self.assertEqual(by_price.name, "VP_10")

        # sort_close bins by price, so the bins are ordered and disjoint and the
        # profile differs from the chronological split.
        self.assertTrue(by_price["mean_close"].is_monotonic_increasing)
        self.assertTrue((by_price["low_close"] <= by_price["mean_close"]).all())
        self.assertTrue((by_price["mean_close"] <= by_price["high_close"]).all())
        self.assertTrue((by_price["high_close"].iloc[:-1].to_numpy() <= by_price["low_close"].iloc[1:].to_numpy()).all())
        self.assertFalse(by_price["total_volume"].equals(chronological["total_volume"]))

        # Both splits account for the same total volume.
        self.assertAlmostEqual(
            float(by_price["total_volume"].sum()),
            float(chronological["total_volume"].sum()),
            places=6,
        )

        # width controls the number of bins on both paths.
        self.assertLessEqual(len(pandas_ta.vp(self.close, self.volume_, width=5, sort_close=True)), 5)

    def test_vfi(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.vfi,
                args=[self.high, self.low, self.close, self.volume_],
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


# ---------------------------------------------------------------------------
# emv divisor defaults to 10000 (tulipy convention)
# ---------------------------------------------------------------------------


class TestEmvDivisorDefault(TestCase):
    """emv divisor defaults to 10000 (tulipy convention post-fix)."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.volume = cls.df["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_emv_default_uses_divisor_10000(self):
        """ta.emv() must match manual calculation with divisor=10000."""
        h, low, v = self.high, self.low, self.volume
        hl_range = (h - low).replace(0, float("nan"))
        midpoint = 0.5 * (h + low)
        distance = midpoint - midpoint.shift(1)
        box_ratio = (v / 10_000) / hl_range
        expected = distance / box_ratio

        result = pandas_ta.emv(h, low, v)
        mask = result.notna() & expected.notna()
        self.assertTrue(mask.any())
        np.testing.assert_allclose(
            result[mask].values,
            expected[mask].values,
            rtol=1e-4,
            err_msg="emv() default divisor must be 10000 (tulipy convention)",
        )

    def test_emv_custom_divisor_differs_from_default(self):
        """emv(divisor=1) must produce a different result than the default."""
        h, low, v = self.high, self.low, self.volume
        default = pandas_ta.emv(h, low, v)
        custom = pandas_ta.emv(h, low, v, divisor=1)

        mask = default.notna() & custom.notna()
        diff = (default[mask] - custom[mask]).abs()
        self.assertTrue(
            diff.max() > 1e-6,
            "emv divisor=1 must differ from the default divisor=10000",
        )


# ---------------------------------------------------------------------------
# ad — None on a missing required input, TypeError on an invalid open_
# ---------------------------------------------------------------------------


class TestAdNoneGuard(TestCase):
    """ad() returns None on a missing required input and rejects an invalid open_."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.close = cls.df["close"]
        cls.volume = cls.df["volume"]

    def test_required_arg_none_returns_none(self):
        """Passing None for required high → None, not AttributeError."""
        result = pandas_ta.ad(high=None, low=self.low, close=self.close, volume=self.volume)
        self.assertIsNone(result)

    def test_invalid_open_list_raises(self):
        """A plain list as open_ is rejected by verify_series (issue #145)."""
        bad_open = list(self.close.values)
        with self.assertRaisesRegex(TypeError, r"ad\(\) expected a pandas Series but got list"):
            pandas_ta.ad(
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
                open_=bad_open,
                talib=False,  # force else-branch where open_ is validated
            )

    def test_valid_call_no_open_returns_series(self):
        """Normal call without open_ returns a Series named 'AD'."""
        result = pandas_ta.ad(high=self.high, low=self.low, close=self.close, volume=self.volume)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "AD")

    def test_valid_call_with_open_returns_series(self):
        """Normal call with valid open_ returns a Series named 'ADo'."""
        result = pandas_ta.ad(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            open_=self.df["open"],
        )
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, "ADo")


# ---------------------------------------------------------------------------
# cmf — an invalid optional open_ raises TypeError instead of crashing
# ---------------------------------------------------------------------------


class TestCmfNoneGuard(TestCase):
    """cmf() rejects an invalid optional open_ with TypeError."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]
        cls.close = cls.df["close"]
        cls.volume = cls.df["volume"]

    def test_invalid_open_list_raises(self):
        """A plain list as open_ is rejected by verify_series (issue #145)."""
        bad_open = list(self.close.values)
        with self.assertRaisesRegex(TypeError, r"cmf\(\) expected a pandas Series but got list"):
            pandas_ta.cmf(
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
                open_=bad_open,
            )

    def test_valid_call_no_open_returns_series(self):
        """Normal call without open_ returns a Series."""
        result = pandas_ta.cmf(high=self.high, low=self.low, close=self.close, volume=self.volume)
        self.assertIsInstance(result, pd.Series)


# ---------------------------------------------------------------------------
# mfi warmup counted an undefined flow
# ---------------------------------------------------------------------------


class TestMfiWarmup(TestCase):
    """The first bar has no previous typical price, so its money flow is undefined.

    mfi counted it as zero, so MFI_14 reported a value at bar 13 from 13 real
    flows; TA-Lib and tulipy start at bar 14.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data().iloc[:200]

    def test_first_value_at_length_plus_drift_minus_one(self):
        for length, drift in ((14, 1), (5, 1), (5, 3)):
            with self.subTest(length=length, drift=drift):
                result = pandas_ta.mfi(self.df.high, self.df.low, self.df.close, self.df.volume, length=length, drift=drift)
                self.assertEqual(self.df.index.get_loc(result.first_valid_index()), length + drift - 1)

    def test_values_after_warmup_unchanged(self):
        d = self.df
        tp = (d.high + d.low + d.close) / 3
        flow = tp * d.volume
        up = flow.where(tp.diff() > 0, 0.0).rolling(14).sum()
        down = flow.where(tp.diff() < 0, 0.0).rolling(14).sum()
        expected = 100 * up / (up + down)
        result = pandas_ta.mfi(d.high, d.low, d.close, d.volume)
        np.testing.assert_allclose(result.iloc[14:], expected.iloc[14:], rtol=1e-12)
