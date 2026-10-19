"""
Tests for DataFrame accessor (df.ta) API consistency — Issue #48.

Covers:
  * The ``indicators()`` method correctly classifies helper methods vs properties.
  * The removed data-fetching API (``df.ta.ticker``, ``ta.yf``, ``ta.av``) is gone.
  * ``prefix``/``suffix`` work as per-call kwargs, not as properties.
  * ``time_range`` accepts valid unit strings and rejects invalid ones.
  * ``df.ta.to_utc`` was removed and raises without touching the frame.
  * ``append=True`` leaves the caller's warning filters unchanged.
  * ``indicators(as_list=True)`` and ``indicators(exclude=[...])`` behave as
    documented.
"""

import inspect
import json
import warnings
from contextlib import redirect_stdout
from datetime import UTC, datetime
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
            "last_run",
            "reverse",
            "time_range",
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

    def test_exchange_removed(self):
        """df.ta.exchange only chose the clock in the last_run string; removed in 0.9.0.

        Reading and setting both raise AttributeError: without the setter, pandas 3
        would put the value on a throwaway accessor and the assignment would do
        nothing. AttributeError, not TypeError, so feature checks keep working.
        """
        before = dict(self.df.attrs)
        for value in ("NYSE", "LSE", None):
            with self.subTest(value=value), self.assertRaisesRegex(AttributeError, r"df.ta.exchange was removed in 0.9.0"):
                self.df.ta.exchange = value
        with self.assertRaisesRegex(AttributeError, r"df.ta.exchange was removed in 0.9.0"):
            _ = self.df.ta.exchange
        self.assertFalse(hasattr(self.df.ta, "exchange"))
        self.assertIsNone(getattr(self.df.ta, "exchange", None))
        self.assertEqual(self.df.attrs, before)

    def test_time_range_unit_persists(self):
        self.df.ta.time_range = "years"
        years = self.df.ta.time_range
        self.df.ta.time_range = "months"
        self.assertGreater(self.df.ta.time_range, years)

    def test_invalid_settings_raise(self):
        """-1, 1.0 and True became cpu_count(); a non-str adjusted became None."""
        for bad in (-1, 1.0, True, "2"):
            with self.assertRaisesRegex(ValueError, r"df.ta.cores must be an integer >= 0 or None"):
                self.df.ta.cores = bad
        self.df.ta.cores = 10_000
        self.assertEqual(self.df.ta.cores, cpu_count())  # capped, as documented
        self.df.ta.cores = None
        self.assertEqual(self.df.ta.cores, 0)  # None resets to the serial default
        with self.assertRaisesRegex(ValueError, r"df.ta.adjusted must be a column name or None, got 5"):
            self.df.ta.adjusted = 5

    def test_settings_do_not_leak_to_other_frames(self):
        self.df.ta.cores = 2
        other = get_sample_data()
        self.assertEqual(other.ta.cores, 0)  # the serial default, not this frame's 2

    def test_accessing_df_ta_does_not_mutate_attrs(self):
        """Touching df.ta must not write into the caller's DataFrame.attrs."""
        before = dict(self.df.attrs)
        _ = self.df.ta
        _ = self.df.ta.cores
        self.assertEqual(self.df.attrs, before)
        self.assertIsNone(self.df.ta.last_run)

    def test_last_run_set_after_an_indicator_runs(self):
        # A UTC timestamp, not the old display string built from the local
        # clock plus a fixed exchange offset.
        before = datetime.now(UTC)
        self.df.ta(kind="sma", length=10)
        last_run = self.df.ta.last_run
        self.assertIs(type(last_run), datetime)  # stdlib, not pd.Timestamp (a subclass)
        self.assertIs(last_run.tzinfo, UTC)
        self.assertTrue(before <= last_run <= datetime.now(UTC))
        self.df.ta.strategy(pandas_ta_classic.CommonStrategy)
        self.assertGreaterEqual(self.df.ta.last_run, last_run)

    def test_last_run_names_a_stamp_it_cannot_read(self):
        # 0.8.32 stored get_time()'s display string; fromisoformat() raised on it without naming last_run.
        for stamp in (
            "Thursday October 17, 2026, NYSE: 7:48:00, Local: 13:48:00 CEST, Day 290/365 (79.0%)",
            datetime.now(UTC),
            "2026-10-17T10:00:00",  # no offset: would come back naive, not UTC
            "2026-10-17T10:00:00+02:00",  # an offset, but not UTC
        ):
            self.df.attrs["_ta_last_run"] = stamp
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, r"df.ta.last_run: .* not a UTC ISO timestamp \(a frame saved by 0.8.x"):
                _ = self.df.ta.last_run
        self.df.ta(kind="sma", length=10)  # overwrites it
        self.assertIs(self.df.ta.last_run.tzinfo, UTC)

    def test_attrs_stay_json_serialisable(self):
        # to_parquet() runs json.dumps(df.attrs), and to_feather() drops attrs it cannot
        # serialise: a datetime stored there broke both after any indicator ran.
        self.df.ta(kind="sma", length=10)
        json.dumps(self.df.attrs)
        self.df.ta.strategy(pandas_ta_classic.CommonStrategy)
        json.dumps(self.df.attrs)

    def test_version_kwarg_removed_and_show_version_validated(self):
        # the 'version' alias of show_version was removed in 0.9.0; the
        # indicator's **kwargs would otherwise swallow it silently
        with self.assertRaisesRegex(TypeError, "no longer accepts 'version'"):
            self.df.ta(kind="sma", length=10, version=True)

        with self.assertRaisesRegex(ValueError, r"show_version must be True or False"):
            self.df.ta(kind="sma", length=10, show_version="yes")

    def test_accessor_raises_for_an_argument_the_frame_cannot_supply(self):
        """These four used to return None with nothing to say what was missing.

        The wrapper filled every required non-column argument with None, so
        df.ta.long_run() ran long_run(fast=None, slow=None) and got None back.
        mavp already raised for its 'periods'; now they all do.
        """
        expected = {
            "long_run": ["'fast'", "'slow'"],
            "short_run": ["'fast'", "'slow'"],
            "tsignals": ["'trend'"],
            "xsignals": ["'signal'", "'xa'", "'xb'"],
            "mavp": ["'periods'"],
        }
        for name, names in expected.items():
            with self.subTest(name=name):
                # ValueError, like every other argument error in the package.
                with self.assertRaises(ValueError) as ctx:
                    getattr(self.df.ta, name)()
                message = str(ctx.exception)
                self.assertIn(f"df.ta.{name}()", message)
                for arg in names:
                    self.assertIn(arg, message)

    def test_accessor_rejects_a_surplus_positional_argument(self):
        """The wrapper bound positionals by hand and dropped any past the last slot.

        df.ta.macd(12, 26, 9, False, 0, 7) ran and ignored the 7, while the direct
        call raises TypeError. Every registered indicator is checked: the error is
        raised before anything is computed, so the sweep is cheap.
        """
        names = sorted({name for names in pandas_ta_classic.Category.values() for name in names})
        for name in names:
            method = getattr(self.df.ta, name)
            slots = [p for p in inspect.signature(method).parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, rf"df\.ta\.{name}\(\) takes .* but {len(slots) + 1} were given"):
                method(*([None] * (len(slots) + 1)))

    def test_accessor_rejects_a_value_given_positionally_and_by_keyword(self):
        with self.assertRaisesRegex(TypeError, r"df\.ta\.sma\(\) got multiple values for argument 'length'"):
            self.df.ta.sma(10, length=20)

    def test_strategy_rejects_surplus_params(self):
        strategy = pandas_ta_classic.Strategy("surplus", [{"kind": "sma", "params": (10, False, 0, 99)}])
        with self.assertRaisesRegex(TypeError, r"df\.ta\.sma\(\) takes at most 3"):
            self.df.ta.strategy(strategy, cores=0)

    def test_accessor_still_serves_those_indicators_when_given_the_argument(self):
        fast = pandas_ta_classic.ema(self.df["close"], length=10)
        slow = pandas_ta_classic.ema(self.df["close"], length=50)
        trend = pandas_ta_classic.above(fast, slow)

        self.assertIsInstance(self.df.ta.long_run(fast=fast, slow=slow), pd.Series)
        self.assertIsInstance(self.df.ta.short_run(fast=fast, slow=slow), pd.Series)
        self.assertEqual(list(self.df.ta.tsignals(trend=trend).columns), ["TS_Trends", "TS_Trades", "TS_Entries", "TS_Exits"])

    def test_show_version_logs_the_version(self):
        with self.assertLogs("pandas_ta_classic.core", level="INFO") as logs:
            self.df.ta(kind="sma", length=10, show_version=True)
        self.assertTrue(any(pandas_ta_classic.version in line for line in logs.output))

    def test_timed_run_attaches_the_elapsed_time(self):
        with self.assertLogs("pandas_ta_classic.core", level="INFO") as logs:
            result = self.df.ta(kind="sma", length=10, timed=True)
        self.assertIsInstance(result.timed, str)
        self.assertIn("ms", result.timed)
        self.assertTrue(any("sma" in line for line in logs.output))

    def test_timed_run_warns_when_the_indicator_returns_none(self):
        # Short input yields an all-NaN result, not None, so use an indicator
        # that really declines: vp has no causal mode and returns None (with a
        # UserWarning of its own) when lookahead=False.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertLogs("pandas_ta_classic.core", level="WARNING") as logs:
                self.assertIsNone(self.df.ta(kind="vp", timed=True, lookahead=False))
        self.assertTrue(any("produced no result" in line for line in logs.output))


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

    def test_time_range_reports_the_real_failure(self):
        # Used to surface as AttributeError: 'int' object has no attribute 'days'
        with self.assertRaisesRegex(TypeError, r"total_time\(\) needs a DatetimeIndex, got RangeIndex"):
            _ = self.df.ta.time_range

    def test_unknown_attribute_still_reports_missing(self):
        with self.assertRaises(AttributeError) as ctx:
            _ = self.df.ta.definitely_not_an_indicator
        self.assertIn("has no attribute 'definitely_not_an_indicator'", str(ctx.exception))


class TestAccessorAppendWarningFilter(TestCase):
    """append=True silences pandas' fragmentation warning only while it appends.

    _append() used to call warnings.simplefilter() bare, which prepended an
    "ignore PerformanceWarning" filter for the rest of the caller's process.
    """

    def test_append_leaves_the_warning_filters_unchanged(self):
        df = get_sample_data().iloc[:100].copy()
        before = list(warnings.filters)
        df.ta.sma(length=10, append=True)
        df.ta.macd(append=True)
        self.assertEqual(list(warnings.filters), before)

    def test_callers_performance_warning_is_still_shown(self):
        df = get_sample_data().iloc[:100].copy()
        df.ta.sma(length=10, append=True)
        # No simplefilter() here: one would sit in front of the leaked filter and hide it.
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("caller's own warning", pd.errors.PerformanceWarning)
        self.assertEqual([str(w.message) for w in caught], ["caller's own warning"])

    def test_fragmentation_warning_stays_silenced_during_strategy(self):
        # 400+ appended columns is far past pandas' 100-block threshold. The
        # filter names PerformanceWarning only: a bare "always" would sit in
        # front of pyproject's error::UserWarning/DeprecationWarning gates.
        def performance_warnings(run):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", pd.errors.PerformanceWarning)
                run()
            return [w for w in caught if issubclass(w.category, pd.errors.PerformanceWarning)]

        # Control: the same kind of column-by-column insertion warns without the silence,
        # so an empty result below means "silenced", not "pandas did not warn".
        plain = get_sample_data().iloc[:300].copy()
        self.assertNotEqual(performance_warnings(lambda: [plain.insert(len(plain.columns), f"x{i}", 0.0) for i in range(120)]), [])

        df = get_sample_data().iloc[:300].copy()
        self.assertEqual(performance_warnings(lambda: df.ta.strategy("all")), [])
        self.assertGreater(len(df.columns), 100)


class TestAccessorToUtcRemoved(TestCase):
    """df.ta.to_utc was a property that converted df's index when it was read.

    hasattr(), dir()-based introspection and IDE autocompletion read it too, so
    merely looking at the accessor changed the caller's frame. It now raises and
    leaves the frame alone; ta.to_utc(df) returns a converted copy.
    """

    def setUp(self):
        self.df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))

    def test_reading_it_raises_and_leaves_the_index_naive(self):
        with self.assertRaisesRegex(AttributeError, r"df\.ta\.to_utc was removed in 0\.9\.0.*ta\.to_utc\(df\)"):
            _ = self.df.ta.to_utc
        self.assertIsNone(self.df.index.tz)

    def test_hasattr_does_not_convert_the_frame(self):
        self.assertFalse(hasattr(self.df.ta, "to_utc"))
        self.assertIsNone(self.df.index.tz)

    def test_the_function_still_converts_a_copy(self):
        utc = pandas_ta_classic.to_utc(self.df)
        self.assertEqual(str(utc.index.tz), "UTC")
        self.assertIsNone(self.df.index.tz)


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

    def test_two_rows_are_enough(self):
        self.assertTrue(self._make_dt_df(["2020-01-01", "2020-01-02"]).ta.datetime_ordered)

    def test_only_duplicate_timestamps_are_not_ordered(self):
        self.assertFalse(self._make_dt_df(["2020-01-01", "2020-01-01"]).ta.datetime_ordered)

    def test_duplicates_at_either_end_are_ordered(self):
        # the first and last labels are what is compared, not their neighbours
        self.assertTrue(self._make_dt_df(["2020-01-01", "2020-01-01", "2020-01-02"]).ta.datetime_ordered)
        self.assertTrue(self._make_dt_df(["2020-01-01", "2020-01-02", "2020-01-02"]).ta.datetime_ordered)

    def test_non_datetime_index_returns_false(self):
        df = pd.DataFrame({"close": [1, 2, 3]}, index=[0, 1, 2])
        self.assertFalse(df.ta.datetime_ordered)
