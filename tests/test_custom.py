import importlib
import os
import shutil
import sys
import tempfile
import types
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
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
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
        # used to log an error and import nothing
        with self.assertRaisesRegex(FileNotFoundError, "xyz_123"):
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

    def test_import_dir_missing_function_raises(self):
        base = os.path.join(self.tmpdir, "missing_func_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_nofunc_"
        module_path = os.path.join(cat_dir, f"{func_name}.py")
        with open(module_path, "w") as f:
            f.write("def some_other_function(): pass\n")

        try:
            with self.assertRaisesRegex(ImportError, func_name):  # used to be logged and skipped
                import_dir(base, verbose=False)
            self.assertFalse(hasattr(pandas_ta_classic, func_name))
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]

    def test_import_dir_broken_module_raises_import_error(self):
        base = os.path.join(self.tmpdir, "broken_module_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_broken_"
        with open(os.path.join(cat_dir, f"{func_name}.py"), "w") as f:
            f.write("raise RuntimeError('boom')\n")

        try:
            # used to call sys.exit(1), killing the caller's interpreter
            with self.assertRaisesRegex(ImportError, func_name) as ctx:
                import_dir(base, verbose=False)
            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]

    def test_import_dir_missing_method_raises(self):
        base = os.path.join(self.tmpdir, "missing_method_dir")
        cat_dir = os.path.join(base, "momentum")
        os.makedirs(cat_dir, exist_ok=True)

        func_name = "_test_cust_nomethod_"
        module_path = os.path.join(cat_dir, f"{func_name}.py")
        with open(module_path, "w") as f:
            f.write(f"def {func_name}(close, **kwargs):\n" f"    return close\n")

        try:
            with self.assertRaisesRegex(ImportError, f"{func_name}_method"):  # used to be logged and skipped
                import_dir(base, verbose=False)
            self.assertFalse(hasattr(pandas_ta_classic, func_name))
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
            if func_name in sys.modules:
                del sys.modules[func_name]

    @staticmethod
    def _write_indicator(cat_dir, name, factor):
        os.makedirs(cat_dir, exist_ok=True)
        with open(os.path.join(cat_dir, f"{name}.py"), "w") as f:
            f.write(
                f"def {name}(close, **kwargs):\n"
                f"    return close * {factor}\n"
                f"def {name}_method(self, **kwargs):\n"
                f"    return {name}(self._get_column('close'))\n"
            )

    @staticmethod
    def _unload(name, *cat_dirs):
        for target in (pandas_ta_classic, AnalysisIndicators):
            if hasattr(target, name):
                delattr(target, name)
        for names in pandas_ta_classic.Category.values():
            if name in names:
                names.remove(name)
        for cat_dir in cat_dirs:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)
        sys.modules.pop(name, None)

    def test_import_dir_rejects_a_stdlib_module_name(self):
        # trend/statistics.py used to reload the stdlib statistics module and then
        # report that it had no function named 'statistics'
        import statistics

        before = dict(vars(statistics))
        cat_dir = os.path.join(self.tmpdir, "stdlib_name_dir", "trend")
        self._write_indicator(cat_dir, "statistics", 2)
        try:
            with self.assertRaisesRegex(ImportError, r"cannot be imported as 'statistics': that name already resolves to .*statistics\.py"):
                import_dir(os.path.dirname(cat_dir), verbose=False)
            self.assertIs(sys.modules["statistics"], statistics)
            self.assertTrue(all(vars(statistics)[k] is v for k, v in before.items()), "the stdlib module was re-executed")
            self.assertNotIn("statistics", pandas_ta_classic.Category["trend"])
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)

    def test_import_dir_rejects_one_name_in_two_categories(self):
        # the second file used to resolve to the first one, so it never loaded
        # and the name was listed under both categories
        name = "_test_cust_twin_"
        base = os.path.join(self.tmpdir, "twin_name_dir")
        cat_dirs = (os.path.join(base, "momentum"), os.path.join(base, "trend"))
        for factor, cat_dir in enumerate(cat_dirs, start=2):
            self._write_indicator(cat_dir, name, factor)
        try:
            with self.assertRaisesRegex(ImportError, rf"cannot be imported as '{name}': that name already resolves to .*{name}\.py"):
                import_dir(base, verbose=False)
            self.assertEqual(sum(name in names for names in pandas_ta_classic.Category.values()), 1)
        finally:
            self._unload(name, *cat_dirs)

    def test_import_dir_rejects_a_name_taken_earlier_on_sys_path_after_the_first_import(self):
        # the check read the cached sys.modules entry, but reload() resolves the
        # name afresh over sys.path, so a same-named file put earlier on sys.path
        # after the first import_dir ran under this indicator's category
        name = "_test_cust_shadow_"
        cat_dir = os.path.join(self.tmpdir, "shadow_dir", "momentum")
        other = os.path.join(self.tmpdir, "shadow_other")
        self._write_indicator(cat_dir, name, 2)
        try:
            import_dir(os.path.dirname(cat_dir), verbose=False)
            self._write_indicator(other, name, 300)
            sys.path.insert(0, other)
            importlib.invalidate_caches()
            with self.assertRaisesRegex(ImportError, rf"cannot be imported as '{name}': that name already resolves to .*shadow_other"):
                import_dir(os.path.dirname(cat_dir), verbose=False)
            self.assertEqual(getattr(pandas_ta_classic, name)(10), 20)
        finally:
            self._unload(name, cat_dir, other)

    def test_import_dir_leaves_sys_path_alone_when_it_refuses_a_name(self):
        cat_dir = os.path.join(self.tmpdir, "stdlib_path_dir", "trend")
        self._write_indicator(cat_dir, "statistics", 2)
        before = list(sys.path)
        with self.assertRaises(ImportError):
            import_dir(os.path.dirname(cat_dir), verbose=False)
        self.assertEqual(sys.path, before)

    def test_import_dir_rejects_a_file_name_that_is_not_a_module_name(self):
        # ab.cd.py imported package 'ab' and raised a bare ModuleNotFoundError
        cat_dir = os.path.join(self.tmpdir, "dotted_dir", "momentum")
        os.makedirs(cat_dir, exist_ok=True)
        with open(os.path.join(cat_dir, "ab.cd.py"), "w") as f:
            f.write("def x(close):\n    return close\n")
        try:
            with self.assertRaisesRegex(ImportError, r"'ab\.cd' is not a valid module name"):
                import_dir(os.path.dirname(cat_dir), verbose=False)
        finally:
            if cat_dir in sys.path:
                sys.path.remove(cat_dir)

    def test_import_dir_reloads_an_edited_indicator(self):
        name = "_test_cust_reload_"
        cat_dir = os.path.join(self.tmpdir, "reload_dir", "momentum")
        self._write_indicator(cat_dir, name, 2)
        try:
            import_dir(os.path.dirname(cat_dir), verbose=False)
            self.assertEqual(getattr(pandas_ta_classic, name)(10), 20)
            self._write_indicator(cat_dir, name, 300)  # a different file size, so a cached .pyc cannot be reused
            importlib.invalidate_caches()
            import_dir(os.path.dirname(cat_dir), verbose=False)
            self.assertEqual(getattr(pandas_ta_classic, name)(10), 3000)
        finally:
            self._unload(name, cat_dir)
