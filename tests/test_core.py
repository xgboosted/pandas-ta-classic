from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta
from pandas_ta_classic.core import Strategy, AnalysisIndicators

from unittest import TestCase
import pandas as pd
from pandas import DataFrame, Series


class TestStrategy(TestCase):
    def test_strategy_creation(self):
        ta = [{"kind": "sma", "length": 10}]
        s = Strategy(name="Test", ta=ta)
        self.assertEqual(s.name, "Test")
        self.assertEqual(s.total_ta(), 1)

    def test_strategy_none_ta(self):
        s = Strategy(name="All", ta=None)
        self.assertEqual(s.total_ta(), 0)

    def test_strategy_empty_ta(self):
        s = Strategy(name="Empty", ta=[])
        self.assertEqual(s.total_ta(), 0)

    def test_strategy_invalid_name(self):
        # name=None triggers the error branch in __post_init__
        s = Strategy(name=None, ta=[{"kind": "sma"}])
        # __post_init__ logs error and returns None (no crash)


class TestAnalysisIndicators(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def test_adjusted_property(self):
        df = self.data.copy()
        df.ta.adjusted = "close"
        self.assertEqual(df.ta.adjusted, "close")
        df.ta.adjusted = None
        self.assertIsNone(df.ta.adjusted)
        df.ta.adjusted = 123
        self.assertIsNone(df.ta.adjusted)

    def test_cores_property(self):
        df = self.data.copy()
        original = df.ta.cores
        df.ta.cores = 1
        self.assertEqual(df.ta.cores, 1)
        df.ta.cores = None
        self.assertGreater(df.ta.cores, 0)
        df.ta.cores = original

    def test_exchange_property(self):
        df = self.data.copy()
        df.ta.exchange = "LSE"
        self.assertEqual(df.ta.exchange, "LSE")
        df.ta.exchange = "INVALID"
        self.assertEqual(df.ta.exchange, "LSE")

    def test_categories_property(self):
        df = self.data.copy()
        cats = df.ta.categories
        self.assertIsInstance(cats, list)
        self.assertIn("momentum", cats)
        self.assertIn("overlap", cats)

    def test_datetime_ordered(self):
        df = self.data.copy()
        self.assertTrue(df.ta.datetime_ordered)

    def test_reverse(self):
        df = self.data.copy()
        rev = df.ta.reverse
        self.assertEqual(rev.index[0], df.index[-1])

    def test_time_range(self):
        df = self.data.copy()
        result = df.ta.time_range
        self.assertIsInstance(result, float)
        df.ta.time_range = "months"
        result2 = df.ta.time_range
        self.assertGreater(result2, result)
        df.ta.time_range = None
        self.assertEqual(df.ta._time_range, "years")

    def test_version(self):
        df = self.data.copy()
        self.assertIsInstance(df.ta.version, str)

    def test_last_run(self):
        df = self.data.copy()
        self.assertIsInstance(df.ta.last_run, str)

    def test_call_kind(self):
        df = self.data.copy()
        result = df.ta(kind="sma")
        self.assertIsInstance(result, Series)

    def test_call_timed(self):
        df = self.data.copy()
        result = df.ta(kind="sma", timed=True)
        self.assertIsInstance(result, Series)
        self.assertTrue(hasattr(result, "timed"))

    def test_call_no_kind(self):
        df = self.data.copy()
        result = df.ta()
        self.assertIsNone(result)

    def test_call_invalid_kind(self):
        df = self.data.copy()
        result = df.ta(kind="nonexistent_indicator_xyz")
        self.assertIsNone(result)

    def test_indicators_as_list(self):
        df = self.data.copy()
        result = df.ta.indicators(as_list=True)
        self.assertIsInstance(result, list)
        self.assertIn("sma", result)

    def test_indicators_exclude(self):
        df = self.data.copy()
        result = df.ta.indicators(as_list=True, exclude=["sma"])
        self.assertNotIn("sma", result)

    def test_indicators_print(self):
        df = self.data.copy()
        result = df.ta.indicators()
        self.assertIsNone(result)

    def test_constants(self):
        df = self.data.copy()
        result = df.ta.constants(True, [42, 100])
        self.assertIn("42", df.columns)
        self.assertIn("100", df.columns)
        df.ta.constants(False, [42, 100])
        self.assertNotIn("42", df.columns)

    def test_dir_includes_indicators(self):
        df = self.data.copy()
        d = dir(df.ta)
        self.assertIn("sma", d)
        self.assertIn("rsi", d)

    def test_check_na_columns(self):
        df = self.data.copy()
        result = df.ta._check_na_columns()
        self.assertIsInstance(result, list)

    def test_get_column_series(self):
        df = self.data.copy()
        result = df.ta._get_column(df["close"])
        pd.testing.assert_series_equal(result, df["close"])

    def test_get_column_str(self):
        df = self.data.copy()
        result = df.ta._get_column("close")
        pd.testing.assert_series_equal(result, df["close"])

    def test_get_column_none(self):
        df = self.data.copy()
        result = df.ta._get_column(None)
        self.assertIsNone(result)

    def test_get_column_str_not_found(self):
        df = self.data.copy()
        result = df.ta._get_column("nonexistent_col_xyz")
        self.assertIsNone(result)

    def test_indicators_by_category(self):
        df = self.data.copy()
        result = df.ta._indicators_by_category("momentum")
        self.assertIsInstance(result, list)
        self.assertIn("rsi", result)

    def test_indicators_by_category_invalid(self):
        df = self.data.copy()
        result = df.ta._indicators_by_category("fake_category")
        self.assertIsNone(result)

    def test_strategy_mode_all(self):
        df = self.data.copy()
        name, mode = df.ta._strategy_mode()
        self.assertEqual(name, "All")
        self.assertTrue(mode["all"])

    def test_strategy_mode_category(self):
        df = self.data.copy()
        name, mode = df.ta._strategy_mode("overlap")
        self.assertEqual(name, "overlap")
        self.assertTrue(mode["category"])

    def test_strategy_mode_strategy_obj(self):
        df = self.data.copy()
        s = Strategy(name="Custom", ta=[{"kind": "sma", "length": 10}])
        name, mode = df.ta._strategy_mode(s)
        self.assertEqual(name, "Custom")
        self.assertTrue(mode["custom"])

    def test_strategy_mode_all_strategy(self):
        df = self.data.copy()
        s = Strategy(name="All", ta=None)
        name, mode = df.ta._strategy_mode(s)
        self.assertTrue(mode["all"])

    def test_strategy_category(self):
        df = self.data.copy()
        df.ta.strategy("statistics", append=True)
        self.assertGreater(len(df.columns), 5)

    def test_strategy_custom(self):
        df = self.data.copy()
        s = Strategy(
            name="Custom",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "ema", "length": 20},
            ],
        )
        df.ta.strategy(s, append=True)
        self.assertGreater(len(df.columns), 5)

    def test_post_process_none(self):
        df = self.data.copy()
        result = df.ta._post_process(None)
        self.assertIsNone(result)

    def test_post_process_verbose_none(self):
        df = self.data.copy()
        result = df.ta._post_process("not_a_dataframe", verbose=True)
        self.assertIsNone(result)

    def test_build_append_fragment_series(self):
        s = Series([1, 2, 3], name="test")
        frag = AnalysisIndicators._build_append_fragment(s)
        self.assertIsInstance(frag, DataFrame)
        self.assertIn("test", frag.columns)

    def test_build_append_fragment_col_names(self):
        s = Series([1, 2, 3], name="test")
        frag = AnalysisIndicators._build_append_fragment(s, col_names=("custom",))
        self.assertIn("custom", frag.columns)

    def test_build_append_fragment_df_col_names(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        frag = AnalysisIndicators._build_append_fragment(df, col_names=("x", "y"))
        self.assertListEqual(list(frag.columns), ["x", "y"])

    def test_build_append_fragment_df_too_few_col_names(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        frag = AnalysisIndicators._build_append_fragment(df, col_names=("x",))
        self.assertIsNone(frag)

    def test_build_append_fragment_str_col_names(self):
        # col_names as string (not tuple) should be auto-wrapped
        s = Series([1, 2, 3], name="test")
        frag = AnalysisIndicators._build_append_fragment(s, col_names="renamed")
        self.assertIsInstance(frag, DataFrame)
        self.assertIn("renamed", frag.columns)

    def test_call_version(self):
        df = self.data.copy()
        df.ta(version=True)

    def test_get_column_adjusted(self):
        df = self.data.copy()
        df.ta.adjusted = "close"
        result = df.ta._get_column(None)
        pd.testing.assert_series_equal(result, df["close"])

    def test_get_column_case_insensitive(self):
        # DataFrame with uppercase column names
        df = self.data.copy()
        df.columns = df.columns.str.upper()
        result = df.ta._get_column("close")
        # Should fuzzy-match "CLOSE"
        self.assertIsNotNone(result)

    def test_post_process_col_numbers(self):
        df = self.data.copy()
        result_df = df.ta(kind="bbands", talib=False)
        # Now test col_numbers selection
        result = df.ta._post_process(result_df, col_numbers=(0, 2))
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(len(result.columns), 2)

    def test_post_process_prefix_suffix(self):
        df = self.data.copy()
        result = df.ta(kind="sma")
        processed = df.ta._post_process(result, prefix="MY", suffix="END")
        self.assertIn("MY_", processed.name)
        self.assertIn("_END", processed.name)

    def test_strategy_verbose_timed(self):
        df = self.data.copy()
        df.ta.strategy("statistics", verbose=True, timed=True, append=True)
        self.assertGreater(len(df.columns), 5)

    def test_strategy_returns(self):
        df = self.data.copy()
        result = df.ta.strategy("statistics", returns=True)
        self.assertIsInstance(result, DataFrame)

    def test_strategy_custom_col_names(self):
        df = self.data.copy()
        s = Strategy(
            name="Custom",
            ta=[
                {"kind": "sma", "length": 10, "col_names": ("MySMA10",)},
            ],
        )
        df.ta.cores = 0
        df.ta.strategy(s, append=True)
        self.assertIn("MySMA10", df.columns)

    def test_strategy_custom_verbose_no_mp(self):
        df = self.data.copy()
        s = Strategy(
            name="Custom",
            ta=[{"kind": "sma", "length": 10}],
        )
        df.ta.cores = 0
        df.ta.strategy(s, verbose=True, append=True)

    def test_strategy_mode_category_strategy_obj(self):
        df = self.data.copy()
        # ta must not be None (that triggers "all" mode)
        s = Strategy(name="statistics", ta=[{"kind": "stdev"}])
        name, mode = df.ta._strategy_mode(s)
        self.assertTrue(mode["category"])

    def test_strategy_custom_length_too_big(self):
        df = self.data.copy()
        s = Strategy(
            name="LengthTest",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "sma", "length": 999999},
            ],
        )
        df.ta.strategy(s, append=True)

    def test_strategy_ordered_false(self):
        df = self.data.copy()
        df.ta.strategy("statistics", ordered=False, append=True)
        self.assertGreater(len(df.columns), 5)

    def test_append_via_accessor(self):
        df = self.data.copy()
        df.ta.sma(length=10, append=True)
        self.assertIn("SMA_10", df.columns)

    def test_accessor_inertia(self):
        df = self.data.copy()
        result = df.ta.inertia()
        self.assertIsInstance(result, Series)

    def test_accessor_inertia_refined(self):
        df = self.data.copy()
        result = df.ta.inertia(refined=True)
        self.assertIsInstance(result, Series)

    def test_accessor_psl(self):
        df = self.data.copy()
        result = df.ta.psl()
        self.assertIsInstance(result, Series)

    def test_accessor_psl_with_open(self):
        df = self.data.copy()
        result = df.ta.psl(open_=df["open"])
        self.assertIsInstance(result, Series)

    def test_accessor_ichimoku(self):
        df = self.data.copy()
        result = df.ta.ichimoku()
        self.assertIsInstance(result, DataFrame)

    def test_accessor_vwap(self):
        df = self.data.copy()
        result = df.ta.vwap()
        self.assertIsInstance(result, Series)

    def test_accessor_psar(self):
        df = self.data.copy()
        result = df.ta.psar()
        self.assertIsInstance(result, DataFrame)

    def test_accessor_tsignals(self):
        df = self.data.copy()
        trend = df["close"] > pandas_ta.sma(df["close"], length=50)
        result = df.ta.tsignals(trend=trend)
        self.assertIsInstance(result, DataFrame)

    def test_accessor_tsignals_none(self):
        df = self.data.copy()
        result = df.ta.tsignals()
        self.assertIsNone(result)

    def test_accessor_xsignals(self):
        df = self.data.copy()
        signal = pandas_ta.rsi(df["close"])
        result = df.ta.xsignals(signal=signal, xa=70, xb=30)
        self.assertIsInstance(result, DataFrame)

    def test_accessor_xsignals_none(self):
        df = self.data.copy()
        result = df.ta.xsignals()
        self.assertIsNone(result)

    def test_accessor_long_run_none(self):
        df = self.data.copy()
        result = df.ta.long_run()
        self.assertIsNone(result)

    def test_accessor_short_run_none(self):
        df = self.data.copy()
        result = df.ta.short_run()
        self.assertIsNone(result)

    def test_accessor_ad(self):
        df = self.data.copy()
        result = df.ta.ad()
        self.assertIsInstance(result, Series)

    def test_accessor_ad_with_open(self):
        df = self.data.copy()
        result = df.ta.ad(open_=df["open"])
        self.assertIsInstance(result, Series)

    def test_accessor_adosc(self):
        df = self.data.copy()
        result = df.ta.adosc()
        self.assertIsInstance(result, Series)

    def test_accessor_adosc_with_open(self):
        df = self.data.copy()
        result = df.ta.adosc(open_=df["open"])
        self.assertIsInstance(result, Series)

    def test_accessor_cmf(self):
        df = self.data.copy()
        result = df.ta.cmf()
        self.assertIsInstance(result, Series)

    def test_accessor_cmf_with_open(self):
        df = self.data.copy()
        result = df.ta.cmf(open_=df["open"])
        self.assertIsInstance(result, Series)

    def test_accessor_above(self):
        df = self.data.copy()
        sma = pandas_ta.sma(df["close"], length=50)
        df["a"] = df["close"]
        df["b"] = sma
        result = df.ta.above()
        self.assertIsInstance(result, Series)

    def test_accessor_below(self):
        df = self.data.copy()
        sma = pandas_ta.sma(df["close"], length=50)
        df["a"] = df["close"]
        df["b"] = sma
        result = df.ta.below()
        self.assertIsInstance(result, Series)

    def test_accessor_above_value(self):
        df = self.data.copy()
        df["a"] = df["close"]
        result = df.ta.above_value(value=100.0)
        self.assertIsInstance(result, Series)

    def test_accessor_below_value(self):
        df = self.data.copy()
        df["a"] = df["close"]
        result = df.ta.below_value(value=100.0)
        self.assertIsInstance(result, Series)

    def test_accessor_cross(self):
        df = self.data.copy()
        sma = pandas_ta.sma(df["close"], length=50)
        df["a"] = df["close"]
        df["b"] = sma
        result = df.ta.cross()
        self.assertIsInstance(result, Series)

    def test_accessor_cross_value(self):
        df = self.data.copy()
        df["a"] = df["close"]
        result = df.ta.cross_value(value=100.0)
        self.assertIsInstance(result, Series)

    def test_strategy_custom_with_params(self):
        df = self.data.copy()
        s = Strategy(
            name="Params",
            ta=[{"kind": "sma", "params": (10,)}],
        )
        df.ta.strategy(s, append=True)

    def test_append_with_pending(self):
        df = self.data.copy()
        # Simulate deferred append mode
        df.ta.pending_appends = []
        result = pandas_ta.sma(df["close"], length=10)
        df.ta._append(result=result, append=True)
        self.assertEqual(len(df.ta.pending_appends), 1)
        df.ta.pending_appends = None
