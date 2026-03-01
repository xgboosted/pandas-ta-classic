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
