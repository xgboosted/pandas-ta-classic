import importlib
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.cycles.msw import _msw_native
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


# ---------------------------------------------------------------------------
# _hilbert NaN handling
# ---------------------------------------------------------------------------


class TestHilbertNanInput(TestCase):
    """The Hilbert Transform loop crashed in int(nan) on any NaN input.

    `dc_period_int = max(int(sp + 0.5), 1)` raised "ValueError: cannot convert
    float NaN to integer" once a NaN reached the smoothed period, which a single
    NaN anywhere in the input guarantees. This contradicted the documented
    contract that an indicator must not crash on a missing bar.
    """

    HT_NAMES = (
        "ht_trendline",
        "ht_dcperiod",
        "ht_dcphase",
        "ht_sine",
        "ht_phasor",
        "ht_trendmode",
    )

    @classmethod
    def setUpClass(cls):
        cls.clean = pd.Series(100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 200)))
        cls.with_nan = cls.clean.copy()
        cls.with_nan.iloc[50] = np.nan

    def test_single_nan_does_not_raise(self):
        """One NaN mid-series returns a result for every ht_* indicator."""
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                result = getattr(pandas_ta, name)(self.with_nan)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, (pd.Series, pd.DataFrame))

    def test_missing_bar_is_skipped(self):
        """The NaN bar reads NaN; every other bar equals the series without it.

        It used to poison the recursion, so every later bar was NaN.
        """
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                result = getattr(pandas_ta, name)(self.with_nan)
                result = result.to_frame() if isinstance(result, pd.Series) else result
                without = getattr(pandas_ta, name)(self.with_nan.drop(index=50))
                without = without.to_frame() if isinstance(without, pd.Series) else without
                self.assertTrue(result.iloc[50].isna().all())
                pd.testing.assert_frame_equal(result.drop(index=50), without, check_dtype=False)

    def test_values_before_the_gap_are_unaffected(self):
        """Bars before the NaN keep the values computed from clean input."""
        clean_result = pandas_ta.ht_trendline(self.clean)
        nan_result = pandas_ta.ht_trendline(self.with_nan)
        mask = clean_result.iloc[:47].notna()
        np.testing.assert_allclose(
            nan_result.iloc[:47][mask].values,
            clean_result.iloc[:47][mask].values,
            rtol=1e-12,
        )

    def test_inf_does_not_raise(self):
        """±Inf is the other half of the documented edge-case contract."""
        with_inf = self.clean.copy()
        with_inf.iloc[50] = np.inf
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                self.assertIsNotNone(getattr(pandas_ta, name)(with_inf))

    def test_clean_input_still_produces_values(self):
        """No NaN anywhere: the guard must not fire."""
        result = pandas_ta.ht_trendline(self.clean)
        self.assertGreater(int(np.isfinite(result).sum()), 100)

    def test_nan_prefix_is_skipped(self):
        """A leading NaN run (chained input) starts the transform after it.

        Without the skip the prefix poisoned the recursion and every ht_*
        output was all-NaN. The result past the prefix must equal the result
        on the series with the prefix removed.
        """
        prefixed = self.clean.copy()
        prefixed.iloc[:10] = np.nan
        trimmed = self.clean.iloc[10:].reset_index(drop=True)
        for name in self.HT_NAMES:
            with self.subTest(indicator=name):
                with_prefix = np.asarray(getattr(pandas_ta, name)(prefixed), dtype=float)[10:]
                without = np.asarray(getattr(pandas_ta, name)(trimmed), dtype=float)
                np.testing.assert_array_equal(with_prefix, without)
                self.assertTrue(np.isfinite(with_prefix).any())


# ---------------------------------------------------------------------------
# ht_* lookback and ht_trendmode undefined bars
# ---------------------------------------------------------------------------


class TestHilbertLookbackAndUndefinedBars(TestCase):
    """ht_* reported unconverged values TA-Lib does not, and ht_trendmode
    turned undefined bars into 0.

    ht_dcperiod returned 0.0 for its first 12 bars and every ht_* emitted
    20-26 values inside TA-Lib's lookback (32 or 63 bars). After a mid-series
    NaN, ht_trendmode converted the NaN it could not compute into 0 -- a
    valid-looking "cycle mode" for every remaining bar. The lookback against
    TA-Lib is checked in tests/oracle/test_oracle_registry.py: by NaN mask in
    test_warmup_matches_talib, and for ht_trendmode, whose lookback reads 0, by
    value in test_native_matches_talib.
    """

    @classmethod
    def setUpClass(cls):
        cls.close = get_sample_data()["close"].iloc[:700].reset_index(drop=True)

    def test_dcperiod_warmup_is_nan_not_zero(self):
        result = pandas_ta.ht_dcperiod(self.close)
        self.assertTrue(result.iloc[:32].isna().all())
        self.assertFalse((result == 0).any())

    def test_trendmode_skips_missing_bar(self):
        """The missing bar reads NaN, not a "cycle mode" 0; later bars keep their values."""
        gapped = self.close.copy()
        gapped.iloc[300] = np.nan
        result = pandas_ta.ht_trendmode(gapped)
        self.assertTrue(np.isnan(result.iloc[300]))
        np.testing.assert_array_equal(result.drop(index=300).to_numpy(float), pandas_ta.ht_trendmode(gapped.drop(index=300)).to_numpy(float))
        self.assertTrue(result.iloc[:300].notna().all())

    def test_trendmode_clean_input_keeps_int_dtype(self):
        self.assertTrue(str(pandas_ta.ht_trendmode(self.close).dtype).startswith("int"))


# msw values depended on whether tulipy was installed
# ---------------------------------------------------------------------------


class TestMswDoesNotDependOnTulipy(TestCase):
    """tulipy's msw truncates pi to 3.1415926; the native loop uses np.pi.

    msw switched to tulipy whenever it was importable, so the same call gave
    values up to 3e-4 apart on two machines. The exact native loop is now the
    default and tulipy is opt-in.
    """

    def test_default_is_the_exact_native_formula(self):
        msw_module = importlib.import_module("pandas_ta_classic.cycles.msw")
        close = get_sample_data().close.iloc[:1000]
        sine, lead = msw_module._msw_native(close.to_numpy(dtype=float), 5)
        result = pandas_ta.msw(close)
        np.testing.assert_array_equal(result.iloc[:, 0].to_numpy(), sine)
        np.testing.assert_array_equal(result.iloc[:, 1].to_numpy(), lead)

    def test_tulipy_flag_is_validated(self):
        with self.assertRaisesRegex(ValueError, r"msw\(\) tulipy must be True or False"):
            pandas_ta.msw(get_sample_data().close.iloc[:100], tulipy="yes")
