"""
Tests for DataFrame accessor (df.ta) API consistency — Issue #48.

Covers:
  * The ``indicators()`` method correctly classifies helper methods vs properties.
  * The removed data-fetching API (``df.ta.ticker``, ``ta.yf``, ``ta.av``) is gone.
  * ``prefix``/``suffix`` work as per-call kwargs, not as properties.
  * ``time_range`` accepts valid unit strings and rejects invalid ones.
  * ``to_utc`` is a property (not callable).
  * ``indicators(as_list=True)`` and ``indicators(exclude=[...])`` behave as
    documented.
"""

from contextlib import redirect_stdout
from io import StringIO
from multiprocessing import cpu_count
from unittest import TestCase, skipIf

import numpy as np
import pandas as pd

import pandas_ta_classic
from tests.config import get_sample_data

# pandas 3 removed accessor caching; pandas 2 still caches df.ta on the
# instance.  Only the premise test below depends on which one is installed.
_PANDAS_MAJOR = int(pd.__version__.split(".")[0])


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
    # Built-in data fetching was removed (deprecated in 0.8.32)
    # ------------------------------------------------------------------

    def test_data_fetching_api_removed(self):
        """df.ta.ticker, ta.yf and ta.av raise AttributeError, not a lookup of an indicator."""
        with self.assertRaises(AttributeError):
            _ = self.df.ta.ticker
        for name in ("yf", "av"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(pandas_ta_classic, name))
                self.assertFalse(hasattr(pandas_ta_classic.utils, name))
        self.assertNotIn("ticker", self._indicator_list())

    # ------------------------------------------------------------------
    # Known helper methods must not appear in the indicator list
    # ------------------------------------------------------------------

    def test_helper_methods_excluded(self):
        """chain, indicators, strategy, unchain must not be in the list."""
        ind = self._indicator_list()
        for name in (
            "chain",
            "indicators",
            "strategy",
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

    # ------------------------------------------------------------------
    # Default call prints to stdout and returns None — Issue #152
    # ------------------------------------------------------------------

    def test_default_call_prints_to_stdout(self):
        """indicators() must print to stdout, not to a logger with a NullHandler."""
        buffer = StringIO()
        with redirect_stdout(buffer):
            result = self.df.ta.indicators()
        output = buffer.getvalue()
        self.assertIsNone(result, "indicators() without as_list=True must return None")
        self.assertIn("Total Indicators & Utilities:", output)
        self.assertIn("sma", output)


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

    def test_invalid_unit_raises(self):
        """An invalid unit used to be stored and then silently computed as years."""
        with self.assertRaisesRegex(ValueError, r"df.ta.time_range must be one of .* got '1y'"):
            self.df.ta.time_range = "1y"
        with self.assertRaisesRegex(ValueError, r"total_time\(\) tf must be one of .* got 'decades'"):
            pandas_ta_classic.utils.total_time(self.df, "decades")

    def test_none_resets_to_years(self):
        self.df.ta.time_range = None
        val = self.df.ta.time_range
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)


class TestAccessorSettablePropertiesPersist(TestCase):
    """Assignments to df.ta.<property> must survive the next df.ta access.

    pandas 3 dropped accessor caching, so ``df.ta`` builds a fresh
    AnalysisIndicators every time.  State kept on the instance is discarded
    the moment it is assigned; it has to live on the DataFrame.

    The requirement is version independent — storing the state on the
    DataFrame is correct whether or not the accessor happens to be cached —
    so every test here runs on both pandas 2 and 3.  Only the premise test
    below observes the caching behaviour itself, and that does differ.
    """

    def setUp(self):
        self.df = get_sample_data()

    @skipIf(_PANDAS_MAJOR < 3, "pandas 2 still caches the accessor on the instance")
    def test_accessor_is_rebuilt_per_access(self):
        """The premise these tests guard against, on pandas 3.

        pandas 2 returns the cached accessor, so ``df.ta is df.ta`` there.
        That is why this one assertion is gated while the persistence tests
        are not: the bug they cover is invisible under caching but the fix
        must hold either way.
        """
        self.assertIsNot(self.df.ta, self.df.ta)

    def test_cores_persists(self):
        self.df.ta.cores = 0
        self.assertEqual(self.df.ta.cores, 0)

    def test_adjusted_persists(self):
        self.df.ta.adjusted = "adj_close"
        self.assertEqual(self.df.ta.adjusted, "adj_close")
        self.df.ta.adjusted = None
        self.assertIsNone(self.df.ta.adjusted)

    def test_exchange_persists(self):
        self.df.ta.exchange = "LSE"
        self.assertEqual(self.df.ta.exchange, "LSE")

    def test_time_range_unit_persists(self):
        self.df.ta.time_range = "years"
        years = self.df.ta.time_range
        self.df.ta.time_range = "months"
        self.assertGreater(self.df.ta.time_range, years)

    def test_invalid_settings_raise(self):
        """-1, 1.0 and True became cpu_count(); an unknown exchange was ignored; a non-str adjusted became None."""
        for bad in (-1, 1.0, True, "2"):
            with self.assertRaisesRegex(ValueError, r"df.ta.cores must be an integer >= 0 or None"):
                self.df.ta.cores = bad
        self.df.ta.cores = 10_000
        self.assertEqual(self.df.ta.cores, cpu_count())  # capped, as documented
        self.df.ta.cores = None
        self.assertEqual(self.df.ta.cores, cpu_count())
        with self.assertRaisesRegex(ValueError, r"df.ta.exchange must be one of .* got 'nope'"):
            self.df.ta.exchange = "nope"
        self.df.ta.exchange = "LSE"
        self.df.ta.exchange = None
        self.assertEqual(self.df.ta.exchange, "NYSE")
        with self.assertRaisesRegex(ValueError, r"df.ta.adjusted must be a column name or None, got 5"):
            self.df.ta.adjusted = 5

    def test_settings_do_not_leak_to_other_frames(self):
        self.df.ta.cores = 0
        self.df.ta.exchange = "LSE"
        other = get_sample_data()
        self.assertNotEqual(other.ta.cores, 0)
        self.assertEqual(other.ta.exchange, "NYSE")

    def test_accessing_df_ta_does_not_mutate_attrs(self):
        """Touching df.ta must not write into the caller's DataFrame.attrs."""
        before = dict(self.df.attrs)
        _ = self.df.ta
        _ = self.df.ta.cores
        self.assertEqual(self.df.attrs, before)
        self.assertIsNone(self.df.ta.last_run)

    def test_last_run_set_after_an_indicator_runs(self):
        self.df.ta(kind="sma", length=10)
        self.assertIsInstance(self.df.ta.last_run, str)

    def test_version_kwarg_deprecated_and_validated(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.df.ta(kind="sma", length=10, version=True)
        self.assertIsNotNone(result)
        self.assertTrue(any("deprecated" in str(x.message) for x in w))

        with self.assertRaisesRegex(ValueError, r"version must be True or False"):
            self.df.ta(kind="sma", length=10, version="yes")

        with self.assertRaisesRegex(ValueError, r"show_version must be True or False"):
            self.df.ta(kind="sma", length=10, show_version="yes")

        with self.assertRaisesRegex(ValueError, r"version must be True or False"):
            self.df.ta(kind="sma", length=10, show_version=True, version="yes")


class TestAccessorNonSeriesColumnArgument(TestCase):
    """Issue #145: df.ta.sma(close=df.close.values) was a silent no-op.

    _get_column returned None for anything that was not a Series or a column
    name, and None is the "argument not given" value, so nothing warned. 0.8.32
    warned; the array now reaches verify_series and raises TypeError.
    """

    def test_ndarray_column_argument_raises(self):
        df = get_sample_data().iloc[:100]
        with self.assertRaisesRegex(TypeError, r"sma\(\) expected a pandas Series but got ndarray"):
            df.ta.sma(length=10, close=df["close"].to_numpy())

    def test_series_column_argument_does_not_warn(self):
        import warnings

        df = get_sample_data().iloc[:100]
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            self.assertEqual(df.ta.sma(length=10, close=df["close"]).notna().sum(), 91)


class TestAccessorAdjustedColumn(TestCase):
    """df.ta.adjusted replaces the default close column, as documented.

    It used to be stored (and, since the pandas 3 fix, persisted) but never
    read: every indicator wrapper passed the literal "close", so the adjusted
    column was ignored.
    """

    def setUp(self):
        self.df = get_sample_data().iloc[:300].copy()
        self.df["adj_close"] = self.df["close"] * 0.5

    def test_adjusted_feeds_default_close(self):
        self.df.ta.adjusted = "adj_close"
        expected = self.df["adj_close"].rolling(10).mean()
        np.testing.assert_allclose(self.df.ta.sma(length=10).to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_explicit_close_overrides_adjusted(self):
        self.df.ta.adjusted = "adj_close"
        expected = self.df["close"].rolling(10).mean()
        np.testing.assert_allclose(self.df.ta.sma(length=10, close="close").to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_resetting_adjusted_restores_close(self):
        self.df.ta.adjusted = "adj_close"
        self.df.ta.adjusted = None
        expected = self.df["close"].rolling(10).mean()
        np.testing.assert_allclose(self.df.ta.sma(length=10).to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_adjusted_leaves_high_and_low_alone(self):
        """Only the close column is adjusted; high/low still come from their columns."""
        self.df.ta.adjusted = "adj_close"
        result = self.df.ta.hl2()
        np.testing.assert_allclose(result.to_numpy(), ((self.df["high"] + self.df["low"]) / 2).to_numpy())


class TestAccessorPropertyErrorsAreNotMasked(TestCase):
    """A failing property must not be reported as a missing attribute.

    An AttributeError raised inside a property getter makes Python fall back
    to __getattr__, which used to answer "no attribute '<name>'" — replacing
    the real cause, with no exception chaining to recover it from.
    """

    def setUp(self):
        # RangeIndex: the time-based properties cannot work on it.
        self.df = pd.DataFrame({"close": [1.0, 2.0]})

    def test_to_utc_reports_the_real_failure(self):
        with self.assertRaises(AttributeError) as ctx:
            _ = self.df.ta.to_utc
        self.assertIn("tz_localize", str(ctx.exception))

    def test_time_range_reports_the_real_failure(self):
        # Used to surface as AttributeError: 'int' object has no attribute 'days'
        with self.assertRaisesRegex(TypeError, r"total_time\(\) needs a DatetimeIndex, got RangeIndex"):
            _ = self.df.ta.time_range

    def test_unknown_attribute_still_reports_missing(self):
        with self.assertRaises(AttributeError) as ctx:
            _ = self.df.ta.definitely_not_an_indicator
        self.assertIn("has no attribute 'definitely_not_an_indicator'", str(ctx.exception))


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
