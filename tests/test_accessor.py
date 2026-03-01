# -*- coding: utf-8 -*-
"""Regression tests for accessor-layer fixes:
- __signature__ on auto-generated methods
- _post_process returns None for insufficient data
- ichimoku accessor returns DataFrame, not tuple
"""
import inspect
from unittest import TestCase

import pandas as pd

from tests.config import get_sample_data
from tests.context import pandas_ta_classic as ta


class TestAccessorSignature(TestCase):
    """Auto-generated accessor methods expose named parameters (no OHLCV)."""

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    def test_sma_signature_has_length(self):
        sig = inspect.signature(self.data.ta.sma)
        self.assertIn("length", sig.parameters)

    def test_sma_signature_no_close(self):
        sig = inspect.signature(self.data.ta.sma)
        self.assertNotIn("close", sig.parameters)

    def test_bbands_signature_no_close(self):
        sig = inspect.signature(self.data.ta.bbands)
        params = sig.parameters
        self.assertNotIn("close", params)
        self.assertIn("length", params)
        self.assertIn("std", params)

    def test_pvr_signature_no_ohlcv(self):
        """pvr takes only OHLCV + no **kwargs — signature should have only self."""
        sig = inspect.signature(self.data.ta.pvr)
        # Only 'self' should remain (no user-facing params).
        non_self = [n for n in sig.parameters if n != "self"]
        self.assertEqual(non_self, [])


class TestAccessorInsufficientData(TestCase):
    """_post_process returns None when indicator produces no result."""

    def test_sma_on_short_df_returns_none(self):
        short = pd.DataFrame({"close": [1.0, 2.0]})
        result = short.ta.sma()  # default length=10, only 2 rows
        self.assertIsNone(result)

    def test_call_with_timed_and_none_result(self):
        """__call__ with timed=True must not crash when result is None."""
        short = pd.DataFrame({"close": [1.0, 2.0]})
        result = short.ta(kind="sma", timed=True)
        self.assertIsNone(result)


class TestAccessorIchimoku(TestCase):
    """Accessor ichimoku returns DataFrame; module-level returns tuple."""

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    def test_accessor_returns_dataframe(self):
        result = self.data.ta.ichimoku()
        self.assertIsInstance(result, pd.DataFrame)

    def test_module_level_returns_tuple(self):
        result = ta.ichimoku(self.data["high"], self.data["low"], self.data["close"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)
