# Must run seperately from the rest of the tests
# in order to successfully run
import os
import sys
import tempfile
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import cpu_count
from time import perf_counter
from unittest import TestCase, mock

from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.core import _custom_kinds
from pandas_ta_classic.custom import import_dir
from tests.config import get_sample_data

# Strategy Testing Parameters
cores = cpu_count()
cumulative = False
speed_table = False
strategy_timed = False
timed = False
verbose = False


class TestStrategyMethods(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.speed_test = DataFrame()
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        cls.speed_test = cls.speed_test.T
        cls.speed_test.index.name = "Test"
        if cls.speed_test.shape[1] == 2:
            cls.speed_test.columns = ["Columns", "Seconds"]
        else:
            # Avoid masking real test failures with a secondary teardown error
            return
        if cumulative:
            cls.speed_test["Cum. Seconds"] = cls.speed_test["Seconds"].cumsum()
        if speed_table:
            cls.speed_test.to_csv("tests/speed_test.csv")
        if timed:
            tca = cls.speed_test["Columns"].sum()
            tcs = cls.speed_test["Seconds"].sum()
            cps = f"[i] Total Columns / Second for All Tests: { tca / tcs:.5f} "
            print("=" * len(cps))
            print(cls.speed_test)
            print(f"[i] Cores: {cls.data.ta.cores}")
            print(f"[i] Total Datapoints per run: {cls.data.shape[0]}")
            print(f"[i] Total Columns added: {tca}")
            print(f"[i] Total Seconds for All Tests: {tcs:.5f}")
            print(cps)
            print("=" * len(cps))
            # tmp = concat([cls.speed_test, cls.speed_test["Columns"].sum(), cls.speed_test["Seconds"].sum()])
            # print(tmp)
        del cls.data

    def setUp(self):
        # Always start with a fresh DataFrame for each test
        self.data = get_sample_data()
        self.added_cols = 0
        self.category = ""
        self.init_cols = len(self.data.columns)
        self.time_diff = 0
        self.result = None
        if verbose:
            print()
        if timed:
            self.stime = perf_counter()

    def tearDown(self):
        if timed:
            self.time_diff = perf_counter() - self.stime
        self.added_cols = len(self.data.columns) - self.init_cols
        self.assertGreaterEqual(self.added_cols, 1)

        self.result = self.data[self.data.columns[-self.added_cols :]]
        self.assertIsInstance(self.result, DataFrame)
        self.data.drop(columns=self.result.columns, inplace=True)

        self.speed_test[self.category] = [self.added_cols, self.time_diff]

    # @skip
    def test_all(self):
        self.category = "All"
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)

    # test_all_ordered was here: it ran the same strategy with ordered=True,
    # which is now the only behaviour, leaving it identical to test_all.

    def test_all_strategy(self):
        self.data.ta.strategy(pandas_ta.AllStrategy, verbose=verbose, timed=strategy_timed)

    def test_all_name_strategy(self):
        self.category = "All"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skipUnless(verbose, "verbose mode only")
    def test_all_multiparams_strategy(self):
        self.category = "All"
        # A broadcast value an indicator cannot use raises (ebsw needs length > 38),
        # so exclude it rather than let the default be substituted silently.
        self.data.ta.strategy(self.category, length=10, exclude=["ebsw"], verbose=verbose, timed=strategy_timed)
        self.data.ta.strategy(self.category, length=50, verbose=verbose, timed=strategy_timed)
        self.data.ta.strategy(self.category, fast=5, slow=10, verbose=verbose, timed=strategy_timed)
        self.category = "All Multiruns with diff Args"  # Rename for Speed Table

    def test_worker_columns_carry_only_reachable_columns(self):
        """Only columns an indicator can reach may be shipped to the Pool.

        The worker copy used to be sliced on the pre-run column count, which on
        a second run already included the first run's indicators.  The oversized
        pickle killed the Windows task pipe with WinError 1450.
        """
        self.category = "Momentum"
        ohlcv = list(self.data.columns)
        self.data["unrelated"] = 1.0

        self.data.ta.strategy(self.category, length=10, verbose=verbose, timed=strategy_timed)
        self.assertGreater(len(self.data.columns), len(ohlcv) + 1)

        # Neither the first run's output nor an unreferenced user column.
        self.assertEqual(self.data.ta._worker_columns([{}]), ohlcv)
        # Unless an indicator is explicitly pointed at it.
        self.assertEqual(self.data.ta._worker_columns([{"close": "unrelated"}]), ohlcv + ["unrelated"])

    # @skip
    def test_candles_category(self):
        self.category = "Candles"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_common(self):
        self.category = "Common"
        self.data.ta.strategy(pandas_ta.CommonStrategy, verbose=verbose, timed=strategy_timed)

    def test_cycles_category(self):
        self.category = "Cycles"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_custom_a(self):
        self.category = "Custom A"

        momo_bands_sma_ta = [
            {"kind": "cdl_pattern", "name": "tristar"},  # 1
            {"kind": "rsi"},  # 1
            {"kind": "macd"},  # 3
            {"kind": "sma", "length": 50},  # 1
            {"kind": "sma", "length": 200},  # 1
            {"kind": "bbands", "length": 20},  # 5 (BBL,BBM,BBU,BBB,BBP)
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"},  # 1
        ]

        custom = pandas_ta.Strategy(
            "Commons with Cumulative Log Return EMA Chain",  # name
            momo_bands_sma_ta,  # ta
            "Common indicators with specific lengths and a chained indicator",  # description
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        # Verify columns were appended (exact count varies by BBANDS version)
        self.assertGreater(len(self.data.columns), 5)
        for col in (
            "CDL_TRISTAR",
            "RSI_14",
            "MACD_12_26_9",
            "SMA_50",
            "SMA_200",
            "CUMLOGRET_1",
            # The chained EMA: it reads CUMLOGRET_1, which an earlier entry
            # produces.  Under the old chunked Pool it silently went missing
            # whenever producer and consumer landed in different chunks, and
            # this check did not cover it.
            "EMA_5_CLR",
        ):
            self.assertIn(col, self.data.columns)

    # @skip
    def test_custom_args_tuple(self):
        self.category = "Custom B"

        custom_args_ta = [
            {"kind": "ema", "params": (5,)},
            {"kind": "fisher", "params": (13, 7)},
        ]

        custom = pandas_ta.Strategy(
            "Custom Args Tuple",
            custom_args_ta,
            "Allow for easy filling in indicator arguments by argument placement.",
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    def test_custom_col_names_tuple(self):
        self.category = "Custom C"

        custom_args_ta = [{"kind": "bbands", "col_names": ("LB", "MB", "UB", "BW", "BP")}]

        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",
            custom_args_ta,
            "Allow for easy renaming of resultant columns",
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_custom_col_numbers_tuple(self):
        self.category = "Custom D"

        custom_args_ta = [{"kind": "macd", "col_numbers": (1,)}]

        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",
            custom_args_ta,
            "Allow for easy selection of resultant columns",
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_custom_e(self):
        self.category = "Custom E"

        amat_logret_ta = [
            {"kind": "amat", "fast": 20, "slow": 50},  # 2
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5},  # 1
        ]

        custom = pandas_ta.Strategy(
            "AMAT Log Returns",  # name
            amat_logret_ta,  # ta
            "AMAT Log Returns",  # description
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        self.data.ta.tsignals(trend=self.data["AMATe_LR_20_50_2"], append=True)
        self.assertEqual(len(self.data.columns), 13)

    # @skip
    def test_momentum_category(self):
        self.category = "Momentum"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_overlap_category(self):
        self.category = "Overlap"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_performance_category(self):
        self.category = "Performance"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_statistics_category(self):
        self.category = "Statistics"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_trend_category(self):
        self.category = "Trend"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_volatility_category(self):
        self.category = "Volatility"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skip
    def test_volume_category(self):
        self.category = "Volume"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # @skipUnless(verbose, "verbose mode only")
    def test_all_no_multiprocessing(self):
        self.category = "All with No Multiprocessing"

        cores = self.data.ta.cores
        self.data.ta.cores = 0
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)
        self.data.ta.cores = cores

    def test_verbose_and_timed_logging(self):
        """verbose/timed only log; nothing else pins those branches.

        The module-level ``verbose``/``strategy_timed`` switches are False, so
        every other strategy test skips them.
        """
        self.category = "Cycles"

        cores = self.data.ta.cores
        self.data.ta.cores = 0
        try:
            with self.assertLogs("pandas_ta_classic.core", level="INFO") as logs:
                self.data.ta.strategy(self.category, verbose=True, timed=True)
        finally:
            self.data.ta.cores = cores

        output = "\n".join(logs.output)
        self.assertIn(f"Strategy: {self.category}", output)
        self.assertIn("Total indicators:", output)
        self.assertIn("Columns added:", output)
        self.assertIn("Last Run:", output)
        self.assertIn("Runtime:", output)

    def test_verbose_reports_excluded_indicators(self):
        """A category run over OHLC-only data drops the volume indicators."""
        self.category = "All"

        no_volume = self.data.drop(columns=["volume"])
        no_volume.ta.cores = 0
        with self.assertLogs("pandas_ta_classic.core", level="INFO") as logs:
            no_volume.ta.strategy(verbose=True)
        self.assertTrue(any("Excluded[" in line for line in logs.output))

        # tearDown measures self.data, so give it something to find.
        self.data.ta.strategy("Cycles", verbose=verbose, timed=strategy_timed)

    # @skipUnless(verbose, "verbose mode only")
    def test_custom_no_multiprocessing(self):
        self.category = "Custom A with No Multiprocessing"

        cores = self.data.ta.cores
        self.data.ta.cores = 0

        momo_bands_sma_ta = [
            {"kind": "rsi"},  # 1
            {"kind": "macd"},  # 3
            {"kind": "sma", "length": 50},  # 1
            {"kind": "sma", "length": 100, "col_names": "sma100"},  # 1
            {"kind": "sma", "length": 200},  # 1
            {"kind": "bbands", "length": 20},  # 3
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"},
        ]

        custom = pandas_ta.Strategy(
            "Commons with Cumulative Log Return EMA Chain",  # name
            momo_bands_sma_ta,  # ta
            "Common indicators with specific lengths and a chained indicator",  # description
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        self.data.ta.cores = cores


class TestStrategyDataclass(TestCase):
    """Unit tests for Strategy dataclass construction and validation."""

    def test_valid_name_and_ta_list(self):
        s = pandas_ta.Strategy("MyStrat", ta=[{"kind": "sma", "length": 10}])
        self.assertEqual(s.name, "MyStrat")
        self.assertIsInstance(s.ta, list)
        self.assertEqual(s.total_ta(), 1)

    def test_empty_ta_list_accepted(self):
        """Strategy(ta=[]) must be accepted (PR #96 behavior change)."""
        s = pandas_ta.Strategy("EmptyStrat", ta=[])
        self.assertEqual(s.name, "EmptyStrat")
        self.assertIsInstance(s.ta, list)
        self.assertEqual(s.total_ta(), 0)

    def test_ta_none_accepted(self):
        """ta=None is valid; used by AllStrategy to mean 'run all'."""
        s = pandas_ta.Strategy("AllStrat", ta=None)
        self.assertIsNone(s.ta)
        self.assertEqual(s.total_ta(), 0)

    def test_invalid_name_raises_value_error(self):
        """Strategy(name=123) must raise ValueError (PR #96 breaking change)."""
        with self.assertRaises(ValueError):
            pandas_ta.Strategy(name=123)

    def test_invalid_ta_type_raises_value_error(self):
        """Non-list ta must raise ValueError."""
        with self.assertRaises(ValueError):
            pandas_ta.Strategy(name="MyStrat", ta="not_a_list")

    def test_created_uses_default_factory(self):
        """Verify created uses field(default_factory=...) so each instance gets its own timestamp."""
        import dataclasses

        field_info = pandas_ta.Strategy.__dataclass_fields__["created"]
        self.assertIsNot(field_info.default_factory, dataclasses.MISSING)
        self.assertIs(field_info.default, dataclasses.MISSING)

    def test_created_is_utc_timestamp(self):
        before = datetime.now(timezone.utc)
        s = pandas_ta.Strategy("TestStrat", ta=[])
        self.assertIs(type(s.created), datetime)  # stdlib, not pd.Timestamp (a subclass)
        self.assertIs(s.created.tzinfo, timezone.utc)
        self.assertTrue(before <= s.created <= datetime.now(timezone.utc))


class TestStrategyNameDoesNotSelect(TestCase):
    """A Strategy's ta list runs whatever its name; only ta=None means every indicator.

    A name equal to a category ("momentum") or to "all" used to discard the
    list and run that whole category, or every indicator, without a word.
    """

    def setUp(self):
        self.data = get_sample_data().iloc[:300]

    def assert_runs_only_its_list(self, name):
        df = self.data.copy()
        df.ta.strategy(pandas_ta.Strategy(name, ta=[{"kind": "rsi"}]))
        self.assertEqual(list(df.columns[len(self.data.columns) :]), ["RSI_14"])

    def test_category_name(self):
        for name in ("momentum", "Overlap"):
            with self.subTest(name=name):
                self.assert_runs_only_its_list(name)

    def test_all_name(self):
        for name in ("all", "All"):
            with self.subTest(name=name):
                self.assert_runs_only_its_list(name)

    def test_empty_list_runs_nothing(self):
        # Strategy("All", ta=[]) used to run every indicator (403 columns on these rows)
        df = self.data.copy()
        df.ta.strategy(pandas_ta.Strategy("All", ta=[]))
        self.assertEqual(list(df.columns), list(self.data.columns))

    def test_missing_input_column_raises_for_a_list(self):
        # all/category mode skips an indicator whose column is missing and reports it;
        # an explicit list asked for that indicator, so the run raises instead. A
        # Strategy named after a category used to take the skipping path.
        df = self.data.drop(columns=["volume"])
        with self.assertRaisesRegex(KeyError, "volume"):
            df.ta.strategy(pandas_ta.Strategy("momentum", ta=[{"kind": "obv"}]))

    def test_ta_none_still_runs_every_indicator(self):
        expected = self.data.copy()
        expected.ta.strategy("all")
        df = self.data.copy()
        df.ta.strategy(pandas_ta.Strategy("momentum", ta=None))
        self.assertEqual(list(df.columns), list(expected.columns))


class TestStrategyWorkerPayload(TestCase):
    """Guards the multiprocessing call site, not just the helper."""

    def test_multiprocessing_ships_the_worker_slice(self):
        """strategy() must pickle only _worker_columns(), not the whole frame.

        Asserting on _worker_columns() alone leaves the call site unguarded:
        replacing the slice with a full copy passed every other test while
        restoring the 1244 MB task payload.
        """
        data = get_sample_data().iloc[:300].copy()
        data["unrelated"] = 1.0
        data.ta.cores = 2
        cls = type(data.ta)
        with mock.patch.object(cls, "_worker_columns", autospec=True, side_effect=cls._worker_columns) as spy:
            data.ta.strategy("Volatility")
        spy.assert_called()
        shipped = spy.side_effect(data.ta, [{}])
        self.assertNotIn("unrelated", shipped)


class TestStrategyDoesNotMutateInputs(TestCase):
    """strategy() must not edit shared or caller-owned lists.

    Category mode removed its exclusions from the global ``Category`` list in
    place, so ``strategy("trend")`` permanently dropped long_run, short_run,
    tsignals and xsignals from ``ta.Category["trend"]``. Custom mode wrote
    ``append: True`` into the caller's dicts and removed too-long entries from
    the caller's ``Strategy.ta``.
    """

    def setUp(self):
        self.data = get_sample_data().iloc[:300].copy()
        self.data.ta.cores = 0

    def test_category_strategy_leaves_category_intact(self):
        # Compare with a fresh discovery, not a snapshot of the live list: an
        # earlier test in the run may already have mutated it, and a snapshot
        # taken after that would pass against the bug.
        from pandas_ta_classic._meta import _build_category_dict

        self.data.ta.strategy("trend")
        self.assertEqual(pandas_ta.Category, _build_category_dict())

    def test_custom_strategy_leaves_strategy_ta_intact(self):
        ta = [{"kind": "sma", "length": 10}, {"kind": "sma", "length": 5000}]
        strategy = pandas_ta.Strategy(name="c", ta=ta)
        self.data.ta.strategy(strategy)
        self.assertEqual(strategy.ta, [{"kind": "sma", "length": 10}, {"kind": "sma", "length": 5000}])
        self.assertIn("SMA_10", self.data.columns)
        # A window longer than the frame gives an all-NaN column, as a direct call does.
        self.assertTrue(self.data["SMA_5000"].isna().all())


class TestCustomStrategyLength(TestCase):
    """Custom mode compared ``kwds["length"] > len(df)`` before running anything.

    ``length=None`` (the library-wide way to ask for the default) raised
    ``TypeError: '>' not supported between instances of 'NoneType' and 'int'``,
    and a window longer than the frame silently dropped the column while
    category mode and direct calls return an all-NaN column.
    """

    def setUp(self):
        self.data = get_sample_data().iloc[:50].copy()
        self.data.ta.cores = 0

    def test_length_none_uses_the_default(self):
        self.data.ta.strategy(pandas_ta.Strategy(name="n", ta=[{"kind": "sma", "length": None}]))
        self.assertIn("SMA_10", self.data.columns)

    def test_invalid_length_raises_from_the_indicator(self):
        with self.assertRaisesRegex(ValueError, r"sma\(\) length must be an integer > 0, got '500'"):
            self.data.ta.strategy(pandas_ta.Strategy(name="s", ta=[{"kind": "sma", "length": "500"}]))

    def test_long_window_matches_category_mode_and_direct_call(self):
        self.data.ta.strategy(pandas_ta.Strategy(name="l", ta=[{"kind": "sma", "length": 500}]))
        direct = self.data.ta.sma(length=500)
        self.assertTrue(self.data["SMA_500"].equals(direct))
        self.assertTrue(self.data["SMA_500"].isna().all())


def _dbl(close, **kwargs):
    out = close * 2
    out.name = "DBL"
    return out


def _dbl_method(self, **kwargs):
    return self._post_process(_dbl(self._get_column("close")), **kwargs)


def _vtw(close, volume, **kwargs):
    out = close * volume
    out.name = "VTW"
    return out


def _vtw_method(self, **kwargs):
    return self._post_process(_vtw(self._get_column("close"), self._get_column("volume")), **kwargs)


class TestCustomIndicatorsNeedSerialStrategy(TestCase):
    """A worker process is a fresh interpreter that never ran custom.import_dir():
    every worker raised AttributeError for a custom indicator the parent had
    accepted, and a custom module reusing a built-in name ran the built-in."""

    def setUp(self):
        self.df = get_sample_data().iloc[:300].copy()
        self.strategy = pandas_ta.Strategy("custom", [{"kind": "sma"}, {"kind": "dbl"}])
        patches = (
            mock.patch.object(pandas_ta.AnalysisIndicators, "dbl", _dbl_method, create=True),
            mock.patch.dict(pandas_ta.Category, {"trend": [*pandas_ta.Category["trend"], "dbl"]}),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_cores_raise_before_any_worker_starts(self):
        with self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\] only serially.*cores=0.*out of the Strategy's 'ta' list"):
            self.df.ta.strategy(self.strategy, cores=2)
        self.assertNotIn("SMA_10", self.df.columns)

    def test_cores_set_on_the_frame_raise_too(self):
        self.df.ta.cores = 2
        with self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\]"):
            self.df.ta.strategy(self.strategy)
        self.assertNotIn("SMA_10", self.df.columns)

    def test_a_category_run_points_to_exclude(self):
        # A category run checks columns against the package function, which import_dir binds too.
        patch = mock.patch.object(pandas_ta, "dbl", _dbl, create=True)
        patch.start()
        self.addCleanup(patch.stop)
        with self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\].*exclude=\[\.\.\.\]"):
            self.df.ta.strategy("trend", cores=2)
        with self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\].*exclude=\[\.\.\.\]"):
            self.df.ta.strategy("all", cores=2)

    def test_a_process_executor_raises_too(self):
        # Nothing is submitted, so the pool never spawns a worker.
        with ProcessPoolExecutor(2) as executor, self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\]"):
            self.df.ta.strategy(self.strategy, executor=executor)

    def test_any_executor_but_a_thread_pool_raises(self):
        class Inline(Executor):
            def submit(self, fn, /, *args, **kwargs):
                future = Future()
                future.set_result(fn(*args, **kwargs))
                return future

        # Whether an executor's workers have the custom methods cannot be checked.
        with self.assertRaisesRegex(ValueError, r"custom indicators \['dbl'\]"):
            self.df.ta.strategy(self.strategy, executor=Inline())

    def test_a_thread_executor_runs_them(self):
        # Threads share this interpreter, custom methods included; this worked before the refusal and still does.
        with ThreadPoolExecutor(2) as executor:
            self.df.ta.strategy(self.strategy, executor=executor)
        self.assertIn("DBL", self.df.columns)
        self.assertTrue(self.df["DBL"].equals(self.df["close"] * 2))

    def test_a_custom_override_of_a_built_in_name_raises(self):
        with (
            mock.patch.object(pandas_ta.AnalysisIndicators, "sma", _dbl_method, create=True),
            self.assertRaisesRegex(ValueError, r"custom indicators \['sma', 'dbl'\]"),
        ):
            self.df.ta.strategy(self.strategy, cores=2)

    def test_a_module_that_only_shares_the_prefix_is_custom(self):
        def ext_method(self, **kwargs):
            return None

        ext_method.__module__ = "pandas_ta_classic_ext"
        with mock.patch.object(pandas_ta.AnalysisIndicators, "ext", ext_method, create=True):
            self.assertEqual(_custom_kinds(["ext", "sma"]), ["ext"])

    def test_serial_strategy_still_runs_them(self):
        self.df.ta.strategy(self.strategy, cores=0)
        self.assertTrue((self.df["DBL"] == self.df["close"] * 2).all())


class TestImportDirWithWorkers(TestCase):
    """The same refusal for an indicator loaded the documented way, through
    custom.import_dir() rather than a patched method."""

    NAME = "_strategy_custom_triple_"

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cat_dir = os.path.join(self.tmpdir, "trend")
        os.makedirs(cat_dir)
        with open(os.path.join(cat_dir, f"{self.NAME}.py"), "w") as f:
            f.write(
                f"def {self.NAME}(close, **kwargs):\n"
                f"    out = close * 3\n"
                f"    out.name = 'TRIPLE'\n"
                f"    return out\n"
                f"def {self.NAME}_method(self, **kwargs):\n"
                f"    return self._post_process({self.NAME}(self._get_column('close')), **kwargs)\n"
            )
        # import_dir appends to Category's lists; a fresh list keeps the real one untouched.
        patch = mock.patch.dict(pandas_ta.Category, {"trend": list(pandas_ta.Category["trend"])})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._unload, cat_dir)
        import_dir(self.tmpdir, verbose=False)
        self.df = get_sample_data().iloc[:300].copy()
        self.strategy = pandas_ta.Strategy("custom", [{"kind": self.NAME}])

    def _unload(self, cat_dir):
        for target in (pandas_ta, pandas_ta.AnalysisIndicators):
            if hasattr(target, self.NAME):
                delattr(target, self.NAME)
        if cat_dir in sys.path:
            sys.path.remove(cat_dir)
        sys.modules.pop(self.NAME, None)

    def test_workers_raise_and_a_serial_run_computes_it(self):
        with self.assertRaisesRegex(ValueError, rf"custom indicators \['{self.NAME}'\]"):
            self.df.ta.strategy(self.strategy, cores=2)
        self.df.ta.strategy(self.strategy, cores=0)
        self.assertTrue((self.df["TRIPLE"] == self.df["close"] * 3).all())


class TestCategoryStrategyWithCustomIndicators(TestCase):
    """custom.import_dir() lists a custom indicator in Category and binds its
    function onto the package, but the column check only asked the package's
    own loader, so strategy("all") and strategy("<category>") raised
    "unknown indicator" as soon as one custom indicator was loaded."""

    def setUp(self):
        self.df = get_sample_data().iloc[:300].copy()
        self.df.ta.cores = 0
        patches = (
            mock.patch.object(pandas_ta, "dbl", _dbl, create=True),
            mock.patch.object(pandas_ta.AnalysisIndicators, "dbl", _dbl_method, create=True),
            mock.patch.object(pandas_ta, "vtw", _vtw, create=True),
            mock.patch.object(pandas_ta.AnalysisIndicators, "vtw", _vtw_method, create=True),
            mock.patch.dict(
                pandas_ta.Category,
                {"trend": [*pandas_ta.Category["trend"], "dbl"], "volume": [*pandas_ta.Category["volume"], "vtw"]},
            ),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_category_and_all_run_custom_indicators(self):
        self.df.ta.strategy("trend")
        self.assertTrue((self.df["DBL"] == self.df["close"] * 2).all())
        df = get_sample_data().iloc[:300].copy()
        df.ta.cores = 0
        df.ta.strategy("all")
        self.assertIn("DBL", df.columns)
        self.assertIn("VTW", df.columns)

    def test_custom_indicator_missing_a_column_is_skipped(self):
        ohlc = self.df[["open", "high", "low", "close"]].copy()
        ohlc.ta.cores = 0
        ohlc.ta.strategy("volume")
        self.assertNotIn("VTW", ohlc.columns)

    def test_a_custom_override_is_checked_with_its_own_signature(self):
        # the built-in sma needs only close; this replacement needs volume too
        with mock.patch.object(pandas_ta, "sma", _vtw):
            ohlc = self.df[["open", "high", "low", "close"]].copy()
            ohlc.ta.cores = 0
            ohlc.ta.strategy("overlap")
        self.assertNotIn("SMA_10", ohlc.columns)
