"""Tests for the lazy-loading infrastructure introduced in PR #125.

Covers:
  * ``_lazy_subpackage`` — ``__getattr__``, ``__setattr__`` unwrapping,
    ``__dir__`` completeness.
  * ``_indicator_loader`` — ``_find_indicator_func`` resolution,
    ``_make_ta_wrapper`` class-level caching.
  * Module-level ``__getattr__`` — top-level indicator access,
    ``cdl_*`` submodule access, removed deprecated names.
  * Regression: cross-package import returns function (not module),
    ``indicators()`` list matches ``Category`` union.
"""

import importlib.metadata
import os
import re
import sys
import types
import unittest
from unittest import mock

import pytest

import pandas_ta_classic
from pandas_ta_classic._indicator_loader import (
    _find_indicator_func,
)
from pandas_ta_classic._meta import _VALID_CATEGORIES, Category
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
        from pandas_ta_classic import trend

        self.assertTrue(
            callable(trend.adx),
            "trend.adx must be a callable function, not a module",
        )
        self.assertIsInstance(mod, types.ModuleType)
        self.assertIsNot(trend.adx, mod)

    def test_dir_returns_known_names(self):
        """__dir__ on a lazy subpackage returns sorted indicator names."""
        from pandas_ta_classic import overlap

        names = dir(overlap)
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 5)
        for name in ("sma", "ema", "wma"):
            self.assertIn(name, names, f"{name!r} must appear in dir(overlap)")

    def test_all_excludes_aliases(self):
        """__all__ must exclude aliases so wildcard imports don't shadow builtins."""
        import pandas_ta_classic.math

        self.assertNotIn("max", pandas_ta_classic.math.__all__)
        self.assertNotIn("min", pandas_ta_classic.math.__all__)
        self.assertNotIn("sum", pandas_ta_classic.math.__all__)
        self.assertIn("rolling_max", pandas_ta_classic.math.__all__)
        self.assertIn("rolling_min", pandas_ta_classic.math.__all__)
        self.assertIn("rolling_sum", pandas_ta_classic.math.__all__)


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

    def test_aliases_not_cached_on_class(self):
        """max/min/sum must NOT be cached as class attributes to prevent builtin shadowing."""
        df = get_sample_data()
        df.ta.max(length=5)
        df.ta.min(length=5)
        df.ta.sum(length=5)

        cls = type(df.ta)
        self.assertNotIn("max", vars(cls), "max must not be cached on AnalysisIndicators")
        self.assertNotIn("min", vars(cls), "min must not be cached on AnalysisIndicators")
        self.assertNotIn("sum", vars(cls), "sum must not be cached on AnalysisIndicators")

    def test_required_column_none_raises(self):
        """Passing close=None to a required column param must raise ValueError."""
        df = get_sample_data()
        with self.assertRaises(ValueError):
            df.ta.rsi(close=None)


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

    def test_cdl_shorthand_access(self):
        """ta.cdl is the shorthand wrapper from candles/cdl_pattern.py.

        It has no submodule of its own and is not a Category entry, so it
        needs its own branch in __getattr__ — it regressed to AttributeError
        when the categories stopped being imported eagerly.
        """
        self.assertTrue(callable(pandas_ta_classic.cdl))
        self.assertIs(pandas_ta_classic.cdl, pandas_ta_classic.candles.cdl)
        self.assertIn("cdl", dir(pandas_ta_classic))

    def test_category_getattr_reloads_after_the_cache_is_cleared(self):
        """The category branch of __getattr__ caches its module on the package.

        Importing a submodule also binds it, so in a normal run the branch never
        executes; dropping the cached attribute forces it.
        """
        for cat in ("momentum", "candles"):
            with self.subTest(cat=cat):
                cached = getattr(pandas_ta_classic, cat)
                delattr(pandas_ta_classic, cat)
                reloaded = getattr(pandas_ta_classic, cat)
                self.assertIs(reloaded, cached)
                # Re-cached, so a second access does not go through __getattr__.
                self.assertIn(cat, vars(pandas_ta_classic))

    def test_cdl_submodule_reraises_a_missing_dependency(self):
        """A cdl_* module that imports something missing must not look absent.

        Swallowing every ModuleNotFoundError turned "this pattern needs a
        package you do not have" into AttributeError.
        """
        import importlib

        real_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "pandas_ta_classic.candles.cdl_2crows":
                raise ModuleNotFoundError("No module named 'not_installed'", name="not_installed")
            return real_import(name, package)

        delattr(pandas_ta_classic, "cdl_2crows")
        try:
            with mock.patch.object(importlib, "import_module", fake_import), self.assertRaises(ModuleNotFoundError):
                _ = pandas_ta_classic.cdl_2crows
        finally:
            self.assertIsInstance(pandas_ta_classic.cdl_2crows, types.ModuleType)

    def test_deprecated_names_removed(self):
        """Names deprecated in 0.8.32 are gone rather than served with a warning."""
        for name in ("CDL_PATTERN_NAMES", "get_time", "EXCHANGE_TZ"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                getattr(pandas_ta_classic, name)
        with self.assertRaises(AttributeError):
            _ = pandas_ta_classic.candles.CDL_PATTERN_NAMES
        self.assertTrue(callable(pandas_ta_classic.utils.get_time))  # internal helper stays

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


class TestCategoryAttrAccess(unittest.TestCase):
    """Regression: `ta.<category>` must resolve deterministically in a fresh
    interpreter, not only after some other indicator has been loaded first.
    """

    def test_all_categories_in_dir(self):
        """Every category subpackage name is advertised by dir().

        They are deliberately absent from __all__, which lists the eagerly
        bound names; the categories resolve through __getattr__ instead,
        exactly like the indicator functions.
        """
        for cat in sorted(_VALID_CATEGORIES):
            self.assertIn(cat, dir(pandas_ta_classic), f"{cat!r} missing from dir(pandas_ta_classic)")

    def test_category_attr_resolves_without_prior_indicator_access(self):
        """`ta.<category>` must be a module in a fresh interpreter, before
        any indicator function has been accessed — regression for the
        order-dependent AttributeError reported against docs/quickstart.md.
        """
        import subprocess
        import sys
        from pathlib import Path

        script = (
            "import sys\n"
            "import types\n"
            "import pandas_ta_classic as ta\n"
            "from pandas_ta_classic._meta import _VALID_CATEGORIES\n"
            # Resolving a category through __getattr__ must not cost anything
            # at import time, so none of them may be loaded yet. performance is
            # exempt: utils/_metrics.py imports from it at module scope, so it
            # loads eagerly for an unrelated reason.
            "for cat in sorted(_VALID_CATEGORIES - {'performance'}):\n"
            "    assert f'pandas_ta_classic.{cat}' not in sys.modules, f'{cat} imported eagerly'\n"
            "for cat in sorted(_VALID_CATEGORIES):\n"
            "    assert hasattr(ta, cat), f'hasattr failed for {cat}'\n"
            "    assert isinstance(getattr(ta, cat), types.ModuleType), cat\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            check=False,  # the return code is asserted below, with stderr as context
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


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


def test_category_discovery_raises_when_it_finds_no_categories():
    """A layout pkgutil cannot list (frozen, sourceless) used to give Category = {}
    without a word, and every indicator lookup failed later; Path.iterdir raised there."""
    from pandas_ta_classic import _meta

    with (
        mock.patch.object(_meta.pkgutil, "iter_modules", return_value=iter(())),
        pytest.raises(ImportError, match=r"could not list the indicator modules of \['candles', 'cycles'"),
    ):
        _meta._build_category_dict()


def _numba_cannot_cache_from_a_zip() -> bool:
    """numba >= 0.62 fails to import an ``@njit(cache=True)`` module from a zip on Windows.

    Its ``ZipCacheLocator`` joins the member path with ``str(Path(...))``, i.e. with
    backslashes on Windows, while zip members use ``/``; since 0.62 the cache stamp
    opens that member and raises KeyError. A numba bug, not one of this package:
    https://github.com/numba/numba/issues/10889
    """
    if sys.platform != "win32" or os.environ.get("NUMBA_DISABLE_JIT") == "1":
        return False
    try:
        version = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        return False
    # "0.63.0rc1" and the like: compare the leading major.minor numbers only
    major, minor = (int(part) for part in re.match(r"(\d+)\.(\d+)", version).groups())
    return (major, minor) >= (0, 62)


@unittest.skipIf(
    _numba_cannot_cache_from_a_zip(),
    "numba >= 0.62 cannot cache an @njit function imported from a zip on Windows (https://github.com/numba/numba/issues/10889)",
)
class TestZipImport(unittest.TestCase):
    def test_package_imports_from_a_zip_archive(self):
        """Category and the candle patterns are discovered through the import
        system: ``Path.iterdir`` / ``os.listdir`` raised FileNotFoundError when
        the package was imported from a zip archive (zipimport)."""
        import json
        import subprocess
        import sys
        import tempfile
        import zipfile
        from pathlib import Path

        from pandas_ta_classic.candles.cdl_pattern import _NATIVE_PATTERNS

        package_dir = Path(pandas_ta_classic.__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp, "pandas_ta_classic.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                for path in package_dir.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        zf.write(path, path.relative_to(package_dir.parent).as_posix())
            script = (
                "import json\n"
                "import pandas_ta_classic as ta\n"
                "from pandas_ta_classic.candles.cdl_pattern import _NATIVE_PATTERNS\n"
                "from tests.config import get_sample_data\n"
                "df = get_sample_data().iloc[:300]\n"
                "patterns = ta.cdl_pattern(df.open, df.high, df.low, df.close, name='all')\n"
                "print(json.dumps([ta.__file__, list(ta.Category.items()), sorted(_NATIVE_PATTERNS), patterns.shape[1]]))\n"
            )
            # The archive comes first on PYTHONPATH, the checkout second for
            # tests.config. cwd is the temp dir: with -c, sys.path[0] is the cwd,
            # and the checkout there would shadow the archive.
            repo_root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": os.pathsep.join([str(archive), str(repo_root)])},
                timeout=300,
                check=False,  # the return code is asserted below, with stderr as context
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        loaded_from, categories, patterns, columns = json.loads(result.stdout)
        self.assertTrue(loaded_from.startswith(str(archive)), loaded_from)  # not the checkout or an editable install
        self.assertEqual(categories, [[k, v] for k, v in Category.items()])
        self.assertEqual(patterns, sorted(_NATIVE_PATTERNS))
        self.assertEqual(columns, 62)
