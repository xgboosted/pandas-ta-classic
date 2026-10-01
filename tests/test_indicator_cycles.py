import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.cycles.msw import _msw_native
from tests.assertions import IndicatorSpec, assert_indicator_standard
from tests.config import get_sample_data


class TestCycles(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    def test_dsp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.dsp,
                args=[self.close],
                expected_name="DSP_14",
                length_override=20,
            ),
        )

    def test_ebsw(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ebsw,
                args=[self.close],
                expected_name="EBSW_40_10",
                length_override=50,
            ),
        )

    def test_ht_dcperiod(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_dcperiod,
                args=[self.close],
                expected_name="HT_DCPERIOD",
            ),
        )

    def test_ht_dcphase(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_dcphase,
                args=[self.close],
                expected_name="HT_DCPHASE",
            ),
        )

    def test_ht_phasor(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_phasor,
                args=[self.close],
                expected_name="HT_PHASOR",
                expected_type=DataFrame,
                expected_columns=["HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"],
            ),
        )

    def test_ht_sine(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_sine,
                args=[self.close],
                expected_name="HT_SINE",
                expected_type=DataFrame,
                expected_columns=["HT_SINE", "HT_LEADSINE"],
            ),
        )

    def test_ht_trendmode(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ht_trendmode,
                args=[self.close],
                expected_name="HT_TRENDMODE",
            ),
        )

    def test_ht_indicators_propagate_an_interior_nan(self):
        """TA-Lib requires NaN-free input; the ports emit NaN instead of crashing.

        The smoothed period feeds its own recursion, so one missing bar makes
        every later value undefined — int(nan) used to raise instead.
        """
        gap = self.close.copy()
        gap.iloc[200] = float("nan")

        funcs = {
            "ht_dcperiod": pandas_ta.ht_dcperiod,
            "ht_dcphase": pandas_ta.ht_dcphase,
            "ht_phasor": pandas_ta.ht_phasor,
            "ht_sine": pandas_ta.ht_sine,
            "ht_trendline": pandas_ta.ht_trendline,
            "ht_trendmode": pandas_ta.ht_trendmode,
        }
        for name, func in funcs.items():
            with self.subTest(name=name):
                clean, poisoned = func(self.close), func(gap)
                self.assertEqual(poisoned.shape, clean.shape)
                # Bars before the gap are untouched. astype(float) because
                # ht_trendmode is int64 without a NaN and float64 with one.
                self.assertTrue(poisoned.iloc[:200].astype(float).equals(clean.iloc[:200].astype(float)))
                # The gap is NaN rather than a crash in int(nan) ...
                self.assertTrue(poisoned.iloc[200].isna().all() if poisoned.ndim > 1 else poisoned.isna().iloc[200])
                # ... and it is the only extra NaN: the recursion carries the
                # previous state forward, so values resume on the next bar.
                self.assertTrue(poisoned.iloc[201:].notna().all().all())
                # They no longer match the clean run, though: the gap perturbed
                # the state the recursion carries.
                self.assertFalse(poisoned.iloc[201:].equals(clean.iloc[201:]))

    def test_msw(self):
        result = pandas_ta.msw(self.close, period=10)
        self.assertIsNotNone(result)

        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.msw,
                args=[self.close],
                expected_name="MSW_5",
                expected_type=DataFrame,
                expected_columns=["MSW_SINE_5", "MSW_LEAD_5"],
            ),
        )

    def test_msw_window_with_a_missing_bar_is_nan_on_both_paths(self):
        # The DFT sums turn NaN, `abs(rp) > 0.001` reads False and the fallback set a
        # phase of +-pi: sine -1.0 for every window holding the gap. Tulip Indicators
        # runs the same loop, so the opt-in tulipy path is faked with the native loop
        # at tulipy's pi, which reproduces it bit for bit.
        def fake_msw(arr, period):
            sine, lead = _msw_native(arr, period, pi=3.1415926)
            return sine[period:], lead[period:]

        close = self.close.iloc[:200].copy()
        close.iloc[100] = np.nan
        with patch.dict(pandas_ta.Imports, {"tulipy": True}), patch.dict(sys.modules, {"tulipy": SimpleNamespace(msw=fake_msw)}):
            tulip = pandas_ta.msw(close, tulipy=True)
        for result in (pandas_ta.msw(close), tulip):
            self.assertTrue(result.iloc[100:105].isna().all().all())
            self.assertTrue(result.iloc[[99, 105]].notna().all().all())
