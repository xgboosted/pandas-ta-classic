import importlib
import os
import sys
import types
import tempfile
import shutil
from unittest import TestCase

import pandas_ta_classic
from pandas_ta_classic import AnalysisIndicators
from pandas_ta_classic.custom import (
    bind,
    create_dir,
    get_module_functions,
    import_dir,
)


class TestCustom(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="pta_custom_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # get_module_functions
    # ------------------------------------------------------------------

    def test_get_module_functions_returns_dict(self):
        rsi_mod = importlib.import_module("pandas_ta_classic.momentum.rsi")
        result = get_module_functions(rsi_mod)
        self.assertIsInstance(result, dict)
        self.assertIn("rsi", result)
        self.assertTrue(callable(result["rsi"]))

    def test_get_module_functions_skips_non_functions(self):
        mod = types.ModuleType("dummy")
        mod.my_func = lambda x: x
        mod.MY_CONST = 42
        mod.my_str = "hello"
        result = get_module_functions(mod)
        self.assertIn("my_func", result)
        self.assertNotIn("MY_CONST", result)
        self.assertNotIn("my_str", result)

    def test_get_module_functions_empty_module(self):
        mod = types.ModuleType("empty")
        self.assertEqual(get_module_functions(mod), {})

    # ------------------------------------------------------------------
    # create_dir
    # ------------------------------------------------------------------

    def test_create_dir_creates_new_directory(self):
        path = os.path.join(self.tmpdir, "new_indicator_dir")
        self.assertFalse(os.path.exists(path))
        create_dir(path, create_categories=False, verbose=False)
        self.assertTrue(os.path.exists(path))

    def test_create_dir_is_idempotent(self):
        path = os.path.join(self.tmpdir, "idempotent_dir")
        create_dir(path, create_categories=False, verbose=False)
        create_dir(path, create_categories=False, verbose=False)
        self.assertTrue(os.path.exists(path))

    def test_create_dir_creates_category_subdirs(self):
        path = os.path.join(self.tmpdir, "with_categories")
        create_dir(path, create_categories=True, verbose=False)
        self.assertTrue(os.path.exists(path))
        categories = [*pandas_ta_classic.Category]
        for cat in categories:
            self.assertTrue(
                os.path.exists(os.path.join(path, cat)),
                f"Missing category subdir: {cat}",
            )

    def test_create_dir_no_categories(self):
        path = os.path.join(self.tmpdir, "no_categories")
        create_dir(path, create_categories=False, verbose=False)
        subdirs = [
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]
        self.assertEqual(subdirs, [])

    # ------------------------------------------------------------------
    # bind
    # ------------------------------------------------------------------

    def test_bind_attaches_to_module_and_class(self):
        def _dummy_func(close):
            return close

        def _dummy_method(self, **kwargs):
            pass

        func_name = "_test_bind_dummy_"
        try:
            bind(func_name, _dummy_func, _dummy_method)
            self.assertTrue(hasattr(pandas_ta_classic, func_name))
            self.assertIs(getattr(pandas_ta_classic, func_name), _dummy_func)
            self.assertTrue(hasattr(AnalysisIndicators, func_name))
        finally:
            if hasattr(pandas_ta_classic, func_name):
                delattr(pandas_ta_classic, func_name)
            if hasattr(AnalysisIndicators, func_name):
                delattr(AnalysisIndicators, func_name)

    # ------------------------------------------------------------------
    # import_dir
    # ------------------------------------------------------------------

    def test_import_dir_nonexistent_path(self):
        # Must return without raising
        import_dir("/nonexistent/path/xyz_123", verbose=False)

    def test_import_dir_skips_invalid_categories(self):
        base = os.path.join(self.tmpdir, "invalid_cat_dir")
        os.makedirs(os.path.join(base, "not_a_category"), exist_ok=True)
        import_dir(base, verbose=False)

    def test_import_dir_loads_valid_indicator(self):
        base = os.path.join(self.tmpdir, "valid_import_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_pta_"
        module_path = os.path.join(cat_dir, f"{func_name}.py")
        with open(module_path, "w") as f:
            f.write(
                f"def {func_name}(close, **kwargs):\n"
                f"    return close\n"
                f"def {func_name}_method(self, **kwargs):\n"
                f"    close = self._get_column(kwargs.pop('close', 'close'))\n"
                f"    return {func_name}(close, **kwargs)\n"
            )

        try:
            import_dir(base, verbose=False)
            self.assertTrue(hasattr(pandas_ta_classic, func_name))
            self.assertIn(func_name, pandas_ta_classic.Category["momentum"])
        finally:
            if hasattr(pandas_ta_classic, func_name):
                delattr(pandas_ta_classic, func_name)
            if hasattr(AnalysisIndicators, func_name):
                delattr(AnalysisIndicators, func_name)
            if func_name in pandas_ta_classic.Category.get("momentum", []):
                pandas_ta_classic.Category["momentum"].remove(func_name)
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]

    def test_import_dir_missing_function_logs_error(self):
        base = os.path.join(self.tmpdir, "missing_func_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_nofunc_"
        module_path = os.path.join(cat_dir, f"{func_name}.py")
        with open(module_path, "w") as f:
            f.write("def some_other_function(): pass\n")

        try:
            import_dir(base, verbose=False)
            self.assertFalse(hasattr(pandas_ta_classic, func_name))
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]

    def test_import_dir_missing_method_logs_error(self):
        base = os.path.join(self.tmpdir, "missing_method_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_nomethod_"
        module_path = os.path.join(cat_dir, f"{func_name}.py")
        with open(module_path, "w") as f:
            f.write(
                f"def {func_name}(close, **kwargs):\n"
                f"    return close\n"
            )

        try:
            import_dir(base, verbose=False)
            self.assertFalse(hasattr(pandas_ta_classic, func_name))
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]
