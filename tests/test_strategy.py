# Must run seperately from the rest of the tests
# in order to successfully run
from multiprocessing import cpu_count
from time import perf_counter

from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame

# Strategy Testing Parameters
cumulative = False
speed_table = False
strategy_timed = False
timed = True
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
        self.data.ta.cores = 0  # Sequential by default; MP tested explicitly
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

    def test_all(self):
        self.category = "All"
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)

    def test_all_ordered(self):
        self.category = "All"
        self.data.ta.strategy(ordered=True, verbose=verbose, timed=strategy_timed)
        self.category = "All Ordered"  # Rename for Speed Table

    def test_all_strategy(self):
        self.data.ta.strategy(
            pandas_ta.AllStrategy, verbose=verbose, timed=strategy_timed
        )

    def test_all_name_strategy(self):
        self.category = "All"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_all_multiparams_strategy(self):
        self.category = "All"
        self.data.ta.strategy(
            self.category, length=10, verbose=verbose, timed=strategy_timed
        )
        self.data.ta.strategy(
            self.category, length=50, verbose=verbose, timed=strategy_timed
        )
        self.data.ta.strategy(
            self.category, fast=5, slow=10, verbose=verbose, timed=strategy_timed
        )
        self.category = "All Multiruns with diff Args"  # Rename for Speed Table

    def test_candles_category(self):
        self.category = "Candles"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_common(self):
        self.category = "Common"
        self.data.ta.strategy(
            pandas_ta.CommonStrategy, verbose=verbose, timed=strategy_timed
        )

    def test_cycles_category(self):
        self.category = "Cycles"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_custom_a(self):
        self.category = "Custom A"
        print()
        print(self.category)

        momo_bands_sma_ta = [
            {"kind": "cdl_pattern", "name": "tristar"},  # 1
            {"kind": "rsi"},  # 1
            {"kind": "macd"},  # 3
            {"kind": "sma", "length": 50},  # 1
            {"kind": "sma", "length": 200},  # 1
            {"kind": "bbands", "length": 20},  # 3
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"},  # 1
        ]

        custom = pandas_ta.Strategy(
            "Commons with Cumulative Log Return EMA Chain",  # name
            momo_bands_sma_ta,  # ta
            "Common indicators with specific lengths and a chained indicator",  # description
        )
        # Chained indicators require sequential execution because later
        # indicators depend on columns produced by earlier ones.
        # Must use a single accessor reference since each .ta access
        # creates a new instance (pandas accessor protocol).
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        self.assertEqual(len(self.data.columns), 19)

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

        custom_args_ta = [
            {"kind": "bbands", "col_names": ("LB", "MB", "UB", "BW", "BP")}
        ]

        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",
            custom_args_ta,
            "Allow for easy renaming of resultant columns",
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    def test_custom_col_numbers_tuple(self):
        self.category = "Custom D"

        custom_args_ta = [{"kind": "macd", "col_numbers": (1,)}]

        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",
            custom_args_ta,
            "Allow for easy selection of resultant columns",
        )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

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
        self.data.ta.strategy(
            custom, verbose=verbose, timed=strategy_timed, ordered=True
        )
        self.data.ta.tsignals(trend=self.data["AMATe_LR_20_50_2"], append=True)
        self.assertEqual(len(self.data.columns), 13)

    def test_momentum_category(self):
        self.category = "Momentum"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_overlap_category(self):
        self.category = "Overlap"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_performance_category(self):
        self.category = "Performance"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_statistics_category(self):
        self.category = "Statistics"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_trend_category(self):
        self.category = "Trend"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_volatility_category(self):
        self.category = "Volatility"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_volume_category(self):
        self.category = "Volume"
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    def test_multiprocessing(self):
        self.category = "Multiprocessing"
        self.data.ta.cores = min(4, cpu_count())
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)
