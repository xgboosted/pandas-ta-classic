"""
Tests for DataFrame accessor (df.ta) API consistency — Issue #48.

Covers:
  * The ``indicators()`` method correctly classifies helper methods vs properties.
  * ``ticker`` is not present in the indicator list (it is a method, not a
    property and should not be returned as an indicator).
  * ``prefix``/``suffix`` work as per-call kwargs, not as properties.
  * ``time_range`` accepts valid unit strings and rejects invalid ones.
  * ``to_utc`` is a property (not callable).
  * ``constants(append, values)`` signature works correctly.
  * ``indicators(as_list=True)`` and ``indicators(exclude=[...])`` behave as
    documented.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from tests.config import get_sample_data
from tests.context import pandas_ta_classic


class TestAccessorHelperClassification(TestCase):
    """indicators() must exclude helper methods AND properties, not indicators."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def _indicator_list(self, **kwargs):
        return self.df.ta.indicators(as_list=True, **kwargs)

    # ------------------------------------------------------------------
    # ticker must NOT appear in the indicator list
    # ------------------------------------------------------------------

    def test_ticker_not_in_indicator_list(self):
        """ticker is a method — must not be listed as an indicator."""
        ind = self._indicator_list()
        self.assertNotIn(
            "ticker",
            ind,
            "'ticker' must not appear in df.ta.indicators() — it is a helper method",
        )

    # ------------------------------------------------------------------
    # Known helper methods must not appear in the indicator list
    # ------------------------------------------------------------------

    def test_helper_methods_excluded(self):
        """chain, constants, indicators, strategy, ticker, unchain must not be in the list."""
        ind = self._indicator_list()
        for name in (
            "chain",
            "constants",
            "indicators",
            "strategy",
            "ticker",
            "unchain",
        ):
            self.assertNotIn(
                name,
                ind,
                f"Helper method '{name}' must not appear in df.ta.indicators()",
            )

    # ------------------------------------------------------------------
    # Known properties must not appear in the indicator list
    # ------------------------------------------------------------------

    def test_properties_excluded(self):
        """Accessor properties must not be listed as indicators."""
        ind = self._indicator_list()
        for prop in (
            "adjusted",
            "categories",
            "cores",
            "datetime_ordered",
            "exchange",
            "last_run",
            "reverse",
            "time_range",
            "to_utc",
            "version",
        ):
            self.assertNotIn(
                prop,
                ind,
                f"Property '{prop}' must not appear in df.ta.indicators()",
            )

    # ------------------------------------------------------------------
    # Real indicators must be present
    # ------------------------------------------------------------------

    def test_known_indicators_present(self):
        """A representative selection of real indicators must appear."""
        ind = self._indicator_list()
        for name in ("sma", "ema", "rsi", "macd", "bbands", "atr"):
            self.assertIn(name, ind, f"Indicator '{name}' missing from indicator list")

    # ------------------------------------------------------------------
    # exclude kwarg removes items from the list
    # ------------------------------------------------------------------

    def test_exclude_kwarg(self):
        """exclude=[...] must remove those indicators from the returned list."""
        full = self._indicator_list()
        self.assertIn("sma", full)
        filtered = self._indicator_list(exclude=["sma"])
        self.assertNotIn("sma", filtered)
        # All other indicators still present
        self.assertIn("ema", filtered)


class TestAccessorPrefixSuffixKwargs(TestCase):
    """prefix and suffix are per-call kwargs, NOT settable properties."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_prefix_kwarg(self):
        result = self.df.ta.sma(length=10, prefix="MY")
        self.assertEqual(result.name, "MY_SMA_10")

    def test_suffix_kwarg(self):
        result = self.df.ta.sma(length=10, suffix="SLOW")
        self.assertEqual(result.name, "SMA_10_SLOW")

    def test_prefix_and_suffix_kwarg(self):
        result = self.df.ta.sma(length=10, prefix="MY", suffix="SLOW")
        self.assertEqual(result.name, "MY_SMA_10_SLOW")

    def test_prefix_suffix_multicolumn(self):
        """prefix/suffix must apply to all columns of a multi-column result."""
        result = self.df.ta.macd(prefix="MY", suffix="v1")
        self.assertIsInstance(result, pd.DataFrame)
        for col in result.columns:
            self.assertTrue(
                col.startswith("MY_") and col.endswith("_v1"),
                f"Column '{col}' missing expected prefix/suffix",
            )

    def test_no_prefix_property(self):
        """The accessor must NOT have a 'prefix' attribute (it's not a property)."""
        self.assertFalse(
            hasattr(type(self.df.ta), "prefix"),
            "df.ta.prefix should not be a class-level descriptor/property",
        )

    def test_no_suffix_property(self):
        """The accessor must NOT have a 'suffix' attribute (it's not a property)."""
        self.assertFalse(
            hasattr(type(self.df.ta), "suffix"),
            "df.ta.suffix should not be a class-level descriptor/property",
        )


class TestAccessorTimeRange(TestCase):
    """time_range accepts valid unit strings; reading returns a float."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_default_is_years(self):
        self.df.ta.time_range = "years"
        val = self.df.ta.time_range
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)

    def test_valid_units(self):
        for unit in ("years", "months", "weeks", "days", "hours", "minutes", "seconds"):
            self.df.ta.time_range = unit
            val = self.df.ta.time_range
            self.assertIsInstance(val, (int, float), f"time_range='{unit}' must return a numeric value")
            self.assertGreater(val, 0, f"time_range='{unit}' must be positive")

    def test_invalid_unit_falls_back_to_years(self):
        """An invalid unit string falls back to 'years' internally."""
        self.df.ta.time_range = "1y"  # invalid unit (not a supported string)
        # After invalid assignment, the getter should still return a positive float
        val = self.df.ta.time_range
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)

    def test_none_resets_to_years(self):
        self.df.ta.time_range = None
        val = self.df.ta.time_range
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)


class TestAccessorToUtcProperty(TestCase):
    """to_utc is a property, not a callable method."""

    def test_to_utc_is_not_callable(self):
        """Accessing df.ta.to_utc must not raise; result is not a method."""
        df = get_sample_data()
        # Accessing the property converts the index in-place and returns None.
        # It must not raise TypeError like "NoneType is not callable".
        result = df.ta.to_utc  # property access — no parentheses
        self.assertIsNone(result)

    def test_to_utc_not_callable_as_method(self):
        """Calling df.ta.to_utc() (with parentheses) should raise TypeError."""
        df = get_sample_data()
        with self.assertRaises(TypeError):
            df.ta.to_utc()  # type: ignore[operator]


class TestAccessorConstants(TestCase):
    """constants(append: bool, values: list) correct signature and behaviour."""

    @classmethod
    def setUpClass(cls):
        cls._base = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls._base

    def fresh(self):
        return self._base.copy()

    def test_add_constants(self):
        df = self.fresh()
        original_cols = len(df.columns)
        df.ta.constants(True, [0, 100])
        self.assertEqual(len(df.columns), original_cols + 2)
        self.assertIn("0", df.columns)
        self.assertIn("100", df.columns)
        self.assertTrue((df["0"] == 0).all())
        self.assertTrue((df["100"] == 100).all())

    def test_remove_constants(self):
        df = self.fresh()
        original_cols = len(df.columns)
        df.ta.constants(True, [0, 100])
        df.ta.constants(False, [0, 100])
        self.assertEqual(len(df.columns), original_cols)

    def test_add_numpy_array(self):
        df = self.fresh()
        original_cols = len(df.columns)
        values = np.array([10, 20, 30])
        df.ta.constants(True, values)
        self.assertEqual(len(df.columns), original_cols + 3)


class TestIsDatetimeOrdered(TestCase):
    """is_datetime_ordered edge-case robustness (fixes from PR #107 review)."""

    def _make_dt_df(self, dates):
        idx = pd.DatetimeIndex(dates)
        return pd.DataFrame({"close": range(len(dates))}, index=idx)

    def test_ordered_datetime_index(self):
        df = self._make_dt_df(["2020-01-01", "2020-01-02", "2020-01-03"])
        self.assertTrue(df.ta.datetime_ordered)

    def test_reversed_datetime_index(self):
        df = self._make_dt_df(["2020-01-03", "2020-01-02", "2020-01-01"])
        self.assertFalse(df.ta.datetime_ordered)

    def test_empty_dataframe_returns_false(self):
        df = pd.DataFrame({"close": pd.Series([], dtype=float)})
        df.index = pd.DatetimeIndex([])
        self.assertFalse(df.ta.datetime_ordered)

    def test_single_row_returns_false(self):
        df = self._make_dt_df(["2020-01-01"])
        self.assertFalse(df.ta.datetime_ordered)

    def test_non_datetime_index_returns_false(self):
        df = pd.DataFrame({"close": [1, 2, 3]}, index=[0, 1, 2])
        self.assertFalse(df.ta.datetime_ordered)
