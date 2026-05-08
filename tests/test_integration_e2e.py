"""Issue #42 — End-to-end integration tests for DataFrame accessor and plugin system.

Covers:
  1. Chaining multiple indicators via append=True incrementally builds the DataFrame.
  2. Custom Strategy execution appends the expected indicator columns.
  3. Custom plugin registration via custom.bind() — indicator callable from df.ta.
  4. import_dir() loads a custom indicator from a temp directory structure and
     binds it to the accessor.
  5. Category-level strategy appends multiple columns.
  6. df.ta(kind=...) dispatch works correctly.

Run:
    python -m unittest tests/test_integration_e2e.py
"""

import os
import sys
import tempfile
import textwrap
from unittest import TestCase

import pandas as pd

import pandas_ta_classic as ta
from pandas_ta_classic import custom
from pandas_ta_classic.core import AnalysisIndicators
from tests.config import get_sample_data
from tests.context import pandas_ta_classic  # noqa: F401 — registers df.ta accessor


# ---------------------------------------------------------------------------
# 1. Indicator chaining via append=True
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


# ---------------------------------------------------------------------------
# 2. Strategy workflow
# ---------------------------------------------------------------------------


class TestCustomStrategyWorkflow(TestCase):
    """End-to-end Strategy workflow using the Strategy dataclass."""

    def setUp(self):
        self.df = get_sample_data()

    def tearDown(self):
        del self.df

    def test_custom_strategy_appends_columns(self):
        """A named custom Strategy appends all configured indicator columns."""
        strategy = ta.Strategy(
            name="e2e_test_strategy",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "ema", "length": 10},
                {"kind": "rsi", "length": 14},
            ],
        )
        self.df.ta.strategy(strategy)
        self.assertIn("SMA_10", self.df.columns)
        self.assertIn("EMA_10", self.df.columns)
        self.assertIn("RSI_14", self.df.columns)

    def test_category_strategy_momentum_appends_columns(self):
        """Running the 'momentum' category strategy appends multiple columns."""
        before = len(self.df.columns)
        self.df.ta.cores = 0  # serial execution — stable in test environment
        self.df.ta.strategy("momentum")
        self.assertGreater(
            len(self.df.columns),
            before,
            "momentum strategy must add at least one column",
        )

    def test_strategy_with_prefix_kwarg(self):
        """prefix kwarg in a strategy indicator entry is honoured."""
        strategy = ta.Strategy(
            name="prefix_test",
            ta=[{"kind": "sma", "length": 10, "prefix": "E2E"}],
        )
        self.df.ta.strategy(strategy)
        prefixed = [c for c in self.df.columns if c.startswith("E2E_")]
        self.assertGreater(len(prefixed), 0, "Prefix kwarg must produce E2E_ columns")

    def test_kind_dispatch_via_call(self):
        """df.ta(kind='sma') dispatches to sma and returns a Series."""
        result = self.df.ta(kind="sma", length=20)
        self.assertIsInstance(result, pd.Series)
        self.assertGreater(result.notna().sum(), 0)

    def test_kind_dispatch_case_insensitive(self):
        """df.ta(kind=...) is case-insensitive (e.g. 'SMA' == 'sma')."""
        lower = self.df.ta(kind="sma", length=10)
        upper = self.df.ta(kind="SMA", length=10)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        pd.testing.assert_series_equal(lower, upper)

    def test_strategy_returns_dataframe_when_requested(self):
        """strategy(returns=True) must return the updated DataFrame."""
        strategy = ta.Strategy(
            name="returns_test",
            ta=[{"kind": "sma", "length": 10}],
        )
        result = self.df.ta.strategy(strategy, returns=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("SMA_10", result.columns)


# ---------------------------------------------------------------------------
# 3. Plugin registration via custom.bind()
# ---------------------------------------------------------------------------


class TestPluginBindRegistration(TestCase):
    """custom.bind() registers a custom indicator accessible from df.ta."""

    _BIND_NAME = "_e2e_test_custom_ind"

    def setUp(self):
        self.df = get_sample_data()

    def tearDown(self):
        if hasattr(ta, self._BIND_NAME):
            delattr(ta, self._BIND_NAME)
        if hasattr(AnalysisIndicators, self._BIND_NAME):
            delattr(AnalysisIndicators, self._BIND_NAME)
        del self.df

    def _make_custom_pair(self):
        """Returns (function, method) for a trivial 'double close' indicator."""
        name = self._BIND_NAME

        def _fn(close, length=None, offset=None, **kwargs):
            result = close * 2.0
            result.name = "DOUBLE_CLOSE"
            return result

        def _method(self_acc, length=None, offset=None, **kwargs):
            close = self_acc._get_column(kwargs.pop("close", "close"))
            result = _fn(close, length=length, offset=offset, **kwargs)
            return self_acc._post_process(result, **kwargs)

        return _fn, _method

    def test_bind_adds_function_to_ta_module(self):
        """After bind(), ta.<name> must be callable."""
        fn, method = self._make_custom_pair()
        custom.bind(self._BIND_NAME, fn, method)
        self.assertTrue(hasattr(ta, self._BIND_NAME))
        self.assertTrue(callable(getattr(ta, self._BIND_NAME)))

    def test_bind_accessor_returns_valid_series(self):
        """After bind(), df.ta.<name>() must return a non-empty Series."""
        fn, method = self._make_custom_pair()
        custom.bind(self._BIND_NAME, fn, method)
        result = getattr(self.df.ta, self._BIND_NAME)()
        self.assertIsInstance(result, pd.Series)
        self.assertGreater(result.notna().sum(), 0)

    def test_bind_result_equals_direct_call(self):
        """df.ta.<name>() must return the same values as calling the function directly."""
        fn, method = self._make_custom_pair()
        custom.bind(self._BIND_NAME, fn, method)

        direct = fn(self.df["close"])
        via_accessor = getattr(self.df.ta, self._BIND_NAME)()

        pd.testing.assert_series_equal(
            direct.reset_index(drop=True),
            via_accessor.reset_index(drop=True),
        )

    def test_bind_append_adds_column(self):
        """After bind(), df.ta.<name>(append=True) appends the column to df."""
        fn, method = self._make_custom_pair()
        custom.bind(self._BIND_NAME, fn, method)

        before = len(self.df.columns)
        getattr(self.df.ta, self._BIND_NAME)(append=True)
        self.assertGreater(len(self.df.columns), before)
        self.assertIn("DOUBLE_CLOSE", self.df.columns)


# ---------------------------------------------------------------------------
# 4. import_dir() loads a custom indicator from a temp directory
# ---------------------------------------------------------------------------


class TestImportDirPlugin(TestCase):
    """import_dir() discovers and binds a custom indicator module from disk."""

    _IND_NAME = "_e2e_import_dir_ind"

    def setUp(self):
        self.df = get_sample_data()
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmppath = self._tmpdir.name

    def tearDown(self):
        # Remove sys.path and global bindings introduced by import_dir
        cat_dir = os.path.join(self._tmppath, "overlap")
        if cat_dir in sys.path:
            sys.path.remove(cat_dir)
        if self._IND_NAME in sys.modules:
            del sys.modules[self._IND_NAME]
        if hasattr(ta, self._IND_NAME):
            delattr(ta, self._IND_NAME)
        if hasattr(AnalysisIndicators, self._IND_NAME):
            delattr(AnalysisIndicators, self._IND_NAME)
        if self._IND_NAME in ta.Category.get("overlap", []):
            ta.Category["overlap"].remove(self._IND_NAME)
        self._tmpdir.cleanup()
        del self.df

    def _write_module(self):
        """Write a minimal indicator .py file to <tmppath>/overlap/."""
        overlap_dir = os.path.join(self._tmppath, "overlap")
        os.makedirs(overlap_dir, exist_ok=True)
        module_path = os.path.join(overlap_dir, f"{self._IND_NAME}.py")
        src = textwrap.dedent(
            f"""\
            from pandas import Series

            def {self._IND_NAME}(close, length=None, offset=None, **kwargs):
                result = close.rolling(2).mean()
                result.name = "IMPORT_DIR_IND"
                return result

            def {self._IND_NAME}_method(self, length=None, offset=None, **kwargs):
                close = self._get_column(kwargs.pop("close", "close"))
                result = {self._IND_NAME}(close, length=length, offset=offset, **kwargs)
                return self._post_process(result, **kwargs)
            """
        )
        with open(module_path, "w") as fh:
            fh.write(src)
        return module_path

    def test_import_dir_binds_function_to_ta(self):
        """After import_dir(), ta.<name> must exist and be callable."""
        self._write_module()
        custom.import_dir(self._tmppath, verbose=False)
        self.assertTrue(
            hasattr(ta, self._IND_NAME),
            f"ta.{self._IND_NAME} must exist after import_dir()",
        )
        self.assertTrue(callable(getattr(ta, self._IND_NAME)))

    def test_import_dir_accessor_method_works(self):
        """After import_dir(), df.ta.<name>() must return a valid Series."""
        self._write_module()
        custom.import_dir(self._tmppath, verbose=False)
        result = getattr(self.df.ta, self._IND_NAME)()
        self.assertIsInstance(result, pd.Series)
        self.assertGreater(result.notna().sum(), 0)

    def test_import_dir_registers_in_category(self):
        """After import_dir(), the indicator must appear in Category['overlap']."""
        self._write_module()
        custom.import_dir(self._tmppath, verbose=False)
        self.assertIn(
            self._IND_NAME,
            ta.Category.get("overlap", []),
            f"{self._IND_NAME} must be in Category['overlap'] after import_dir()",
        )

    def test_import_dir_skips_invalid_category_dir(self):
        """import_dir() must silently skip subdirectories that are not valid categories."""
        # Create a non-category subdirectory alongside the valid one
        bad_dir = os.path.join(self._tmppath, "not_a_category")
        os.makedirs(bad_dir, exist_ok=True)
        self._write_module()
        # Must not raise
        custom.import_dir(self._tmppath, verbose=False)
        # The indicator in 'overlap' must still be bound
        self.assertTrue(hasattr(ta, self._IND_NAME))
