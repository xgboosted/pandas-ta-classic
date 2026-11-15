from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


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

    def test_marketfi(self):
        self.assertIsNone(pandas_ta.marketfi(self.high, None, self.volume_))
        self.assertIsNone(pandas_ta.marketfi(self.high, self.low, None))

    def test_pvr(self):
        result = pandas_ta.pvr(self.close, self.volume_)
        # sample indicator values from SPY
        # Bar 0 has no difference to rank; it read 1, a rising close, before.
        self.assertTrue(result.isna().iloc[0])
        self.assertEqual(result.iloc[1], 3)
        self.assertEqual(result.iloc[4], 2)
        self.assertEqual(result.iloc[6], 4)

    def test_vp_fill(self):
        # vp's rows are price bins, so contracts/test_index_contract.py leaves it out
        base = pandas_ta.vp(self.close, self.volume_)
        pd.testing.assert_frame_equal(pandas_ta.vp(self.close, self.volume_, fillna=0), base.fillna(0))
        pd.testing.assert_frame_equal(pandas_ta.vp(self.close, self.volume_, fill_method="ffill"), base.ffill())
        pd.testing.assert_frame_equal(pandas_ta.vp(self.close, self.volume_, fill_method="bfill"), base.bfill())

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

    def test_wad(self):
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
