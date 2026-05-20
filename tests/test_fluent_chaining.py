"""
Tests for fluent API chaining (Issue #36).

Covers:
  * ``df.ta.chain()`` activates chain mode and returns the DataFrame.
  * Chaining indicators: ``df.ta.chain().sma(10).ta.rsi(14).ta.macd()``
  * ``df.ta.unchain()`` deactivates chain mode.
  * Chain mode auto-appends results to the DataFrame.
  * Chain mode with ``append=False`` (results appended anyway).
  * Nested / re-entrant chain calls are idempotent.
  * Chain mode does not leak between DataFrames.
"""

from unittest import TestCase

import pandas as pd

from tests.config import get_sample_data
from tests.context import pandas_ta_classic  # noqa: F401 — registers df.ta accessor


class TestFluentChaining(TestCase):
    """Fluent API chaining for df.ta (Issue #36)."""

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        self.df = self.data.copy()

    def tearDown(self):
        del self.df

    # ------------------------------------------------------------------
    # chain() / unchain() basics
    # ------------------------------------------------------------------

    def test_chain_returns_dataframe(self):
        """chain() returns the AnalysisIndicators accessor."""
        result = self.df.ta.chain()
        # chain() returns the accessor, not a DataFrame
        self.assertTrue(hasattr(result, 'sma'))

    def test_unchain_returns_dataframe(self):
        """unchain() returns the DataFrame (so .ta is available for normal calls)."""
        self.df.ta.chain()
        result = self.df.ta.unchain()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIs(result, self.df)

    def test_chain_sets_flag(self):
        """chain() stores the flag in df.attrs."""
        self.df.ta.chain()
        self.assertTrue(self.df.attrs.get("_ta_chain"))

    def test_unchain_clears_flag(self):
        """unchain() removes the flag from df.attrs."""
        self.df.ta.chain()
        self.df.ta.unchain()
        self.assertNotIn("_ta_chain", self.df.attrs)

    # ------------------------------------------------------------------
    # Fluent chaining
    # ------------------------------------------------------------------

    def test_chain_single_indicator(self):
        """Single indicator in chain mode appends and returns DataFrame."""
        col_count = len(self.df.columns)
        # chain() returns accessor → .sma(10) returns DataFrame
        result = self.df.ta.chain().sma(10)
        self.assertIsInstance(result, pd.DataFrame)
        # Column added
        self.assertGreater(len(self.df.columns), col_count)
        self.assertIn("SMA_10", self.df.columns)

    def test_chain_multiple_indicators(self):
        """Chain sma -> rsi -> macd in one expression."""
        col_count = len(self.df.columns)
        # chain().sma() returns df → .ta accesses new chain-aware accessor
        result = self.df.ta.chain().sma(10).ta.rsi(14).ta.macd()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(self.df.columns), col_count)
        # Verify all indicators were appended
        self.assertIn("SMA_10", self.df.columns)
        self.assertIn("RSI_14", self.df.columns)
        self.assertIn("MACD_12_26_9", self.df.columns)

    def test_chain_with_bbands(self):
        """Chain a multi-column indicator (BBANDS)."""
        result = self.df.ta.chain().bbands(20)
        self.assertIsInstance(result, pd.DataFrame)
        for col in ("BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"):
            self.assertIn(col, self.df.columns, f"{col} not appended")

    def test_chain_still_computes_correct_values(self):
        """Chained RSI should match non-chained RSI."""
        # Non-chained
        normal = self.df.ta.rsi(14)
        # Chained (need fresh copy)
        df2 = self.data.copy()
        df2.ta.chain().rsi(14)
        chained_result = df2["RSI_14"]
        pd.testing.assert_series_equal(normal, chained_result, check_names=False)

    # ------------------------------------------------------------------
    # Chain mode isolation
    # ------------------------------------------------------------------

    def test_chain_does_not_leak_across_dataframes(self):
        """Chain mode on one df should not affect another."""
        df2 = self.data.copy()
        self.df.ta.chain()
        self.assertTrue(self.df.attrs.get("_ta_chain"))
        self.assertNotIn("_ta_chain", df2.attrs)

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def test_double_chain_is_idempotent(self):
        """Calling chain() twice is a no-op the second time."""
        self.df.ta.chain()
        self.df.ta.chain()
        # Still in chain mode
        self.assertTrue(self.df.attrs.get("_ta_chain"))

    # ------------------------------------------------------------------
    # Chain with explicit kwargs
    # ------------------------------------------------------------------

    def test_chain_with_prefix(self):
        """Prefix works in chain mode."""
        self.df.ta.chain().sma(10, prefix="MY")
        self.assertIn("MY_SMA_10", self.df.columns)

    def test_chain_with_col_names(self):
        """col_names works in chain mode (multi-column result)."""
        self.df.ta.chain().bbands(20, col_names=("LOW", "MID", "UP", "BW", "PCT"))
        self.assertIn("LOW", self.df.columns)
        self.assertIn("MID", self.df.columns)
        self.assertIn("UP", self.df.columns)
        self.assertIn("BW", self.df.columns)
        self.assertIn("PCT", self.df.columns)
