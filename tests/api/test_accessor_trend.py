import math
from unittest import TestCase

from pandas import DataFrame

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase
from tests.config import get_sample_data


class TestTrendExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_long_run_ext(self):
        # Without fast/slow Series the indicator cannot compute. It used to
        # return None, which named nothing; the accessor now says which
        # arguments the DataFrame cannot supply.
        with self.assertRaisesRegex(ValueError, r"df\.ta\.long_run\(\) requires 'fast', 'slow'"):
            self.data.ta.long_run(append=True)

        fast = self.data.ta.ema(8)
        slow = self.data.ta.ema(21)
        self.data.ta.long_run(fast, slow, append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LR_2")

    def test_short_run_ext(self):
        # See test_long_run_ext: the accessor names the missing arguments.
        with self.assertRaisesRegex(ValueError, r"df\.ta\.short_run\(\) requires 'fast', 'slow'"):
            self.data.ta.short_run(append=True)

        fast = self.data.ta.ema(8)
        slow = self.data.ta.ema(21)
        self.data.ta.short_run(fast, slow, append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "SR_2")


class TestExtTrendDeeper(AccessorValueCase):
    def test_adx_ext(self):
        df = self.fresh()
        df.ta.adx(append=True)
        self._check_appended(df, "ADX_14")
        valid = df["ADX_14"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(), "ADX out of [0,100]")

    def test_aroon_ext(self):
        df = self.fresh()
        df.ta.aroon(append=True)
        aroon_cols = [c for c in df.columns if c.startswith("AROON")]
        self.assertTrue(len(aroon_cols) >= 2, "Expected at least 2 Aroon columns")
        for col in aroon_cols:
            self._check_appended(df, col)

    def test_decay_ext(self):
        df = self.fresh()
        df.ta.decay(append=True)
        self._check_last_col(df)

    def test_psar_ext(self):
        df = self.fresh()
        df.ta.psar(append=True)
        psar_cols = [c for c in df.columns if c.startswith("PSAR")]
        self.assertTrue(len(psar_cols) >= 1, "No PSAR columns appended")
        for col in psar_cols:
            # PSARr column is binary (0/1); PSARl/PSARs are price levels or NaN
            valid = df[col].dropna()
            if not valid.empty:
                self.assertTrue(
                    all(math.isfinite(v) for v in valid),
                    f"PSAR column {col!r} contains non-finite values",
                )

    def test_supertrend_ext(self):
        df = self.fresh()
        df.ta.supertrend(append=True)
        self._check_appended(df, "SUPERT_7_3.0")

    def test_vortex_ext(self):
        df = self.fresh()
        df.ta.vortex(append=True)
        self._check_appended(df, "VTXP_14")
        self._check_appended(df, "VTXM_14")
