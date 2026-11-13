"""The comparison helpers in tests/assertions.py must fail on wrong results.

A helper that accepts everything turns every test built on it into a no-op:
``assert_talib`` used to fall back to a correlation threshold, which a result
scaled by 2 or shifted by a constant passes with correlation 1.0.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from tests.assertions import assert_talib


class TestAssertTalib(TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range("2020-01-01", periods=50, freq="D")
        values = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 50))
        values[:5] = np.nan
        cls.expected = pd.Series(values, index=index)

    def test_accepts_rounding(self):
        assert_talib(self, self.expected * (1 + 1e-12), self.expected)

    def test_rejects_a_scaled_result(self):
        with self.assertRaises(AssertionError):
            assert_talib(self, self.expected * (1 + 1e-6), self.expected)

    def test_rejects_a_constant_offset(self):
        with self.assertRaises(AssertionError):
            assert_talib(self, self.expected + 1e-6, self.expected)

    def test_rejects_a_different_warmup(self):
        with self.assertRaises(AssertionError):
            assert_talib(self, self.expected.fillna(100.0), self.expected)

    def test_rejects_a_missing_column(self):
        frame = pd.DataFrame({"a": self.expected, "b": self.expected})
        with self.assertRaises(AssertionError):
            assert_talib(self, frame[["a"]], frame)
