"""
Regression tests for known bugs.

Each test documents a specific bug (with issue reference or description) and
guards against future regressions. Tests are named after the bug they cover.
"""

import unittest

import pandas as pd

from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta


class TestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = get_sample_data()
        data.columns = data.columns.str.lower()
        cls.close = data["close"]


class TestBbandsRegressions(TestSetup):
    """
    Regression tests for Bollinger Bands (bbands).
    """

    def test_bbands_offset_does_not_corrupt_bbp(self):
        """
        Bug: bbands(close, offset=1) returned bandwidth values in the BBP column
        instead of percent values, because the offset was applied before the
        DataFrame was assembled — causing each column to pick up the wrong
        pre-shifted series.

        Fix: offset must be applied to each named Series independently after
        all band computations are complete.
        """
        result_0 = pandas_ta.bbands(self.close, talib=False)
        result_1 = pandas_ta.bbands(self.close, offset=1, talib=False)

        # BBP with offset=1 must be shift(1) of BBP without offset
        pd.testing.assert_series_equal(
            result_1["BBP_5_2.0"],
            result_0["BBP_5_2.0"].shift(1),
            check_names=False,
        )

    def test_bbands_offset_does_not_corrupt_bbb(self):
        """
        Companion to the BBP offset bug: BBB (bandwidth) must also shift
        correctly and must not contain percent values after offset is applied.
        """
        result_0 = pandas_ta.bbands(self.close, talib=False)
        result_1 = pandas_ta.bbands(self.close, offset=1, talib=False)

        pd.testing.assert_series_equal(
            result_1["BBB_5_2.0"],
            result_0["BBB_5_2.0"].shift(1),
            check_names=False,
        )

    def test_bbands_bbb_is_positive(self):
        """BBB (bandwidth) must always be non-negative."""
        result = pandas_ta.bbands(self.close, talib=False)
        self.assertTrue(result["BBB_5_2.0"].dropna().ge(0).all())

    def test_bbands_bbp_is_between_zero_and_one(self):
        """
        BBP (percent B) is defined as (close - lower) / (upper - lower).
        For a well-behaved price series it stays in [0, 1].
        """
        result = pandas_ta.bbands(self.close, talib=False)
        self.assertTrue(result["BBP_5_2.0"].dropna().between(0, 1).all())

    def test_bbands_has_five_columns(self):
        """bbands must return exactly 5 columns: BBL, BBM, BBU, BBB, BBP."""
        result = pandas_ta.bbands(self.close, talib=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns),
            ["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "BBB_5_2.0", "BBP_5_2.0"],
        )

    def test_bbands_upper_above_mid_above_lower(self):
        """Upper band >= middle band >= lower band at all non-NaN rows."""
        result = pandas_ta.bbands(self.close, talib=False).dropna()
        self.assertTrue((result["BBU_5_2.0"] >= result["BBM_5_2.0"]).all())
        self.assertTrue((result["BBM_5_2.0"] >= result["BBL_5_2.0"]).all())


if __name__ == "__main__":
    unittest.main()
