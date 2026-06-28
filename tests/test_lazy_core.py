"""Tests for the lazy-loading infrastructure introduced in PR #125.

Covers:
  * ``_lazy_subpackage`` — ``__getattr__``, ``__setattr__`` unwrapping,
    ``__dir__`` completeness.
  * ``_indicator_loader`` — ``_find_indicator_func`` resolution,
    ``_make_ta_wrapper`` class-level caching.
  * Module-level ``__getattr__`` — top-level indicator access,
    ``cdl_*`` submodule access, ``CDL_PATTERN_NAMES`` deprecation.
  * Regression: cross-package import returns function (not module),
    ``indicators()`` list matches ``Category`` union.
"""

import types
import unittest
import warnings

import pandas_ta_classic

from pandas_ta_classic._indicator_loader import (
    _find_indicator_func,
)
from pandas_ta_classic._meta import Category

from tests.config import get_sample_data

# ---------------------------------------------------------------------------
# _lazy_subpackage tests
# ---------------------------------------------------------------------------


class TestLazySubpackage(unittest.TestCase):
    """Verify _LazySubpackage __getattr__ / __setattr__ / __dir__ behaviour."""

    def test_getattr_loads_indicator_func(self):
        """Accessing a known name on a lazy subpackage returns a callable function."""
        import pandas_ta_classic.momentum as mom

        func = mom.rsi
        self.assertTrue(callable(func), "mom.rsi must be callable")
        self.assertIsNotNone(func)

    def test_getattr_unknown_raises_attrerror(self):
        """Accessing an unknown name on a lazy subpackage raises AttributeError."""
        import pandas_ta_classic.volatility as vol

        with self.assertRaises(AttributeError):
            _ = vol.__nonexistent_indicator_xyz__

    def test_setattr_unwraps_module_to_func(self):
        """Submodule import through a lazy subpackage unwraps to the function.

        Regression test: the old wildcard-import pattern could leave a
        submodule bound as the attribute instead of the function.
        """
        import importlib

        # Force-import a submodule inside a lazy subpackage
        mod = importlib.import_module("pandas_ta_classic.trend.adx")

        # The parent package's attr must be the *function*, not the module
        import pandas_ta_classic.trend as trend

        self.assertTrue(
            callable(trend.adx),
            "trend.adx must be a callable function, not a module",
        )
        self.assertIsInstance(mod, types.ModuleType)
        self.assertIsNot(trend.adx, mod)

    def test_dir_returns_known_names(self):
        """__dir__ on a lazy subpackage returns sorted indicator names."""
        import pandas_ta_classic.overlap as overlap

        names = dir(overlap)
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 5)
        for name in ("sma", "ema", "wma"):
            self.assertIn(name, names, f"{name!r} must appear in dir(overlap)")


# ---------------------------------------------------------------------------
# _indicator_loader tests
# ---------------------------------------------------------------------------


class TestIndicatorLoader(unittest.TestCase):
    def test_find_indicator_func_returns_callable(self):
        func = _find_indicator_func("rsi")
        self.assertTrue(callable(func), "_find_indicator_func('rsi') must be callable")
        self.assertEqual(func.__name__, "rsi")

    def test_find_indicator_func_unknown_returns_none(self):
        self.assertIsNone(_find_indicator_func("__nonexistent_xyz__"))

    def test_find_indicator_func_math_alias_resolves(self):
        """max/min/sum math aliases resolve to rolling_max/rolling_min/rolling_sum."""
        for alias, canonical in [("max", "rolling_max"), ("min", "rolling_min"), ("sum", "rolling_sum")]:
            func = _find_indicator_func(alias)
            self.assertTrue(callable(func), f"alias {alias!r} must resolve to callable")
            self.assertEqual(func.__name__, canonical)

    def test_make_ta_wrapper_caches_on_class(self):
        """After a first access through __getattr__, the wrapper is cached on class."""
        df = get_sample_data()

        # First access goes through __getattr__ and caches on AnalysisIndicators
        result = df.ta.sma(length=10)
        self.assertIsNotNone(result)

        cls = type(df.ta)
        self.assertTrue(
            hasattr(cls, "sma"),
            "After first access, 'sma' must be cached on AnalysisIndicators class",
        )

        # A second DataFrame instance should use the cached wrapper
        df2 = get_sample_data()
        result2 = df2.ta.sma(length=10)
        self.assertIsNotNone(result2)
        self.assertEqual(len(result), len(result2))


# ---------------------------------------------------------------------------
# Module-level __getattr__ tests
# ---------------------------------------------------------------------------


class TestModuleGetattr(unittest.TestCase):
    def test_lazy_load_indicator_from_module(self):
        """pandas_ta_classic.rsi must return a callable, not a module."""
        func = pandas_ta_classic.rsi
        self.assertTrue(callable(func))
        self.assertFalse(isinstance(func, types.ModuleType))
        self.assertEqual(func.__name__, "rsi")

    def test_lazy_load_after_first_access(self):
        """Second access to same indicator returns the cached function."""
        func1 = pandas_ta_classic.sma
        func2 = pandas_ta_classic.sma
        self.assertIs(func1, func2, "Repeated access must return the same cached object")

    def test_cdl_submodule_access(self):
        """cdl_* names not in _CANDLE_TOP_LEVEL must return submodules."""
        mod = pandas_ta_classic.cdl_2crows
        self.assertIsInstance(mod, types.ModuleType)
        self.assertTrue(hasattr(mod, "cdl_2crows"), "submodule must contain its pattern function")

    def test_cdl_pattern_names_deprecation(self):
        """Accessing CDL_PATTERN_NAMES emits DeprecationWarning, returns ALL_PATTERNS."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pandas_ta_classic.CDL_PATTERN_NAMES
            self.assertEqual(result, pandas_ta_classic.ALL_PATTERNS)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(
                len(deprecation_warnings),
                1,
                "CDL_PATTERN_NAMES must emit exactly one DeprecationWarning",
            )
            self.assertIn("deprecated", str(deprecation_warnings[0].message).lower())
            self.assertIn("ALL_PATTERNS", str(deprecation_warnings[0].message))


# ---------------------------------------------------------------------------
# __dir__ completeness
# ---------------------------------------------------------------------------


class TestDirCompleteness(unittest.TestCase):
    def test_dir_includes_all_category_indicators(self):
        """dir(pandas_ta_classic) must include every indicator from Category."""
        all_category_indicators = {ind for inds in Category.values() for ind in inds}
        all_category_indicators.add("ALL_PATTERNS")
        module_names = set(dir(pandas_ta_classic))
        missing = all_category_indicators - module_names
        self.assertEqual(
            missing,
            set(),
            f"dir(pandas_ta_classic) missing indicators: {missing}",
        )

    def test_dir_sorted_and_contains_public_api(self):
        names = dir(pandas_ta_classic)
        self.assertEqual(names, sorted(names), "dir() must return sorted list")
        self.assertIn("rsi", names)
        self.assertIn("sma", names)
        self.assertIn("ALL_PATTERNS", names)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.df

    def test_cross_pkg_import_returns_func_volatility_atr(self):
        """from pandas_ta_classic.volatility import atr → callable (not module)."""
        from pandas_ta_classic.volatility import atr

        self.assertTrue(callable(atr), "volatility.atr import must return callable")
        self.assertFalse(isinstance(atr, types.ModuleType), "volatility.atr must not be a module")

    def test_cross_pkg_import_returns_func_trend_adx(self):
        """from pandas_ta_classic.trend import adx → callable (not module)."""
        from pandas_ta_classic.trend import adx

        self.assertTrue(callable(adx), "trend.adx import must return callable")
        self.assertFalse(isinstance(adx, types.ModuleType), "trend.adx must not be a module")

    def test_cross_pkg_import_returns_func_statistics_stdev(self):
        """from pandas_ta_classic.statistics import stdev → callable (not module)."""
        from pandas_ta_classic.statistics import stdev

        self.assertTrue(callable(stdev), "statistics.stdev import must return callable")
        self.assertFalse(isinstance(stdev, types.ModuleType), "statistics.stdev must not be a module")

    def test_cross_pkg_import_math_alias_max(self):
        """from pandas_ta_classic.math import max → callable (rolling_max)."""
        from pandas_ta_classic.math import max as math_max

        self.assertTrue(callable(math_max), "math.max alias import must return callable")
        self.assertEqual(math_max.__name__, "rolling_max")

    def test_indicators_list_matches_category(self):
        """indicators(as_list=True) union (minus helpers) matches Category union."""
        indicator_list = set(self.df.ta.indicators(as_list=True))

        category_indicators = {ind for inds in Category.values() for ind in inds}

        # indicators() excludes some by default (above, below, cross, etc.)
        # but the full set should be a superset of category indicators
        # after accounting for built-in exclusions.
        builtin_excluded = {
            "above",
            "above_value",
            "below",
            "below_value",
            "cross",
            "cross_value",
            "long_run",
            "short_run",
            "td_seq",
            "tsignals",
            "vp",
            "xsignals",
        }
        expected = category_indicators - builtin_excluded

        missing = expected - indicator_list
        self.assertEqual(
            missing,
            set(),
            f"indicators() missing Category indicators (after exclusions): {missing}",
        )

    def test_df_ta_accessor_categories_match_meta(self):
        """df.ta.categories must match the Category dict keys."""
        accessor_cats = set(self.df.ta.categories)
        meta_cats = set(Category.keys())
        self.assertEqual(accessor_cats, meta_cats)
