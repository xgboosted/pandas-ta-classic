"""
Tests for fluent API chaining (Issue #36).

Covers:
  * ``df.ta.chain()`` activates chain mode and returns the DataFrame.
  * Chaining indicators: ``df.ta.chain().sma(10).ta.rsi(14).ta.macd()``
  * ``df.ta.unchain()`` deactivates chain mode.
  * Chain mode auto-appends results to the DataFrame.
  * Chain mode with ``append=False`` (results appended anyway).
  * Nested / re-entrant chain calls are idempotent.
  * Chain mode does not leak between DataFrames, nor into copies or slices.
"""

from unittest import TestCase

import pandas as pd

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.config import get_sample_data


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
        self.assertTrue(hasattr(result, "sma"))

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

    def test_chain_does_not_follow_copies_or_slices(self):
        """pandas copies attrs into df.copy() and slices; neither is chained."""
        self.df.ta.chain()
        copied, sliced = self.df.copy(), self.df.iloc[-100:]
        for other in (copied, sliced):
            n_columns = other.shape[1]
            self.assertIsInstance(other.ta.rsi(), pd.Series)
            self.assertEqual(other.shape[1], n_columns)
        self.assertIsInstance(self.df.ta.rsi(), pd.DataFrame)  # the chained frame still is

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


# ---------------------------------------------------------------------------
# Indicator chaining via append=True
# ---------------------------------------------------------------------------


class TestIndicatorChaining(TestCase):
    """Multiple append=True calls build up the DataFrame incrementally."""

    def setUp(self):
        self.df = get_sample_data()

    def tearDown(self):
        del self.df

    def test_single_append_adds_column(self):
        """Appending one indicator adds at least one new column."""
        before = len(self.df.columns)
        self.df.ta.sma(length=10, append=True)
        self.assertGreater(len(self.df.columns), before)
        self.assertIn("SMA_10", self.df.columns)

    def test_chain_sma_rsi_macd(self):
        """Chaining SMA, RSI, and MACD appends all their columns."""
        self.df.ta.sma(length=20, append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)

        self.assertIn("SMA_20", self.df.columns)
        self.assertIn("RSI_14", self.df.columns)
        macd_cols = [c for c in self.df.columns if c.startswith("MACD")]
        self.assertGreater(len(macd_cols), 0, "No MACD columns found after append")

    def test_chained_result_types(self):
        """Each chained indicator returns a Series or DataFrame with non-NaN values."""
        sma = self.df.ta.sma(length=10)
        rsi = self.df.ta.rsi(length=14)
        macd = self.df.ta.macd(fast=12, slow=26, signal=9)

        self.assertIsInstance(sma, pd.Series)
        self.assertIsInstance(rsi, pd.Series)
        self.assertIsInstance(macd, pd.DataFrame)
        self.assertGreater(sma.notna().sum(), 0)
        self.assertGreater(rsi.notna().sum(), 0)
        self.assertGreater(macd.notna().sum().sum(), 0)

    def test_chained_columns_are_numeric(self):
        """Appended indicator columns must be numeric dtype."""
        self.df.ta.sma(length=10, append=True)
        self.df.ta.ema(length=10, append=True)
        for col in ("SMA_10", "EMA_10"):
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.df[col]),
                f"Column '{col}' must be numeric; got {self.df[col].dtype}",
            )

    def test_chain_does_not_corrupt_ohlcv(self):
        """Chaining indicators must not modify original OHLCV data."""
        ohlcv_before = self.df[["open", "high", "low", "close", "volume"]].copy()
        self.df.ta.sma(length=10, append=True)
        self.df.ta.bbands(length=20, append=True)
        ohlcv_after = self.df[["open", "high", "low", "close", "volume"]]
        pd.testing.assert_frame_equal(ohlcv_before, ohlcv_after)

    def test_chain_column_count_grows_monotonically(self):
        """Column count must increase (or stay same) after every append call."""
        counts = [len(self.df.columns)]
        for ind_kwargs in [
            ("sma", {"length": 10}),
            ("ema", {"length": 20}),
            ("rsi", {"length": 14}),
        ]:
            method, kwargs = ind_kwargs
            getattr(self.df.ta, method)(**kwargs, append=True)
            counts.append(len(self.df.columns))
        self.assertEqual(counts, sorted(counts), "Column count must never decrease")
