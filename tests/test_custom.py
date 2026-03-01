import os
import sys
import textwrap
import tempfile
import types
from unittest import TestCase

import pandas_ta_classic
import pandas_ta_classic.custom as custom


class TestCustom(TestCase):

    def test_create_dir_with_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "indicators")
            custom.create_dir(path, verbose=False)
            self.assertTrue(os.path.isdir(path))
            for cat in list(pandas_ta_classic.Category.keys())[:3]:
                self.assertTrue(os.path.isdir(os.path.join(path, cat)))

    def test_create_dir_without_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "indicators")
            custom.create_dir(path, create_categories=False, verbose=False)
            self.assertTrue(os.path.isdir(path))

    def test_create_dir_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "indicators")
            os.makedirs(path)
            # Calling again on existing dir should not raise
            custom.create_dir(path, verbose=False)
            self.assertTrue(os.path.isdir(path))

    def test_get_module_functions(self):
        mod = types.ModuleType("_test_mod")

        def my_fn():
            pass

        mod.my_fn = my_fn
        mod.NOT_A_FN = 42
        result = custom.get_module_functions(mod)
        self.assertIn("my_fn", result)
        self.assertEqual(result["my_fn"], my_fn)
        self.assertNotIn("NOT_A_FN", result)

    def test_import_dir_missing_path(self):
        # Covers early-return branch when path doesn't exist
        custom.import_dir("/nonexistent/path/xyz_abc_123", verbose=False)

    def test_import_dir_invalid_category(self):
        # Covers "skipping non-category dir" branch
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "not_a_valid_category"))
            custom.import_dir(tmp, verbose=False)

    def test_import_dir_valid_indicator(self):
        # Ensure cleanup runs even if the test fails
        def _cleanup_custom_indicator():
            from pandas_ta_classic import AnalysisIndicators

            for attr in ("my_custom_ind", "my_custom_ind_method"):
                if hasattr(pandas_ta_classic, attr):
                    delattr(pandas_ta_classic, attr)
                if hasattr(AnalysisIndicators, attr):
                    delattr(AnalysisIndicators, attr)
            cat = pandas_ta_classic.Category.get("momentum", [])
            if "my_custom_ind" in cat:
                cat.remove("my_custom_ind")

        self.addCleanup(_cleanup_custom_indicator)

        # Covers full load + bind path
        with tempfile.TemporaryDirectory() as tmp:
            cat_dir = os.path.join(tmp, "momentum")
            os.makedirs(cat_dir)
            indicator_src = textwrap.dedent(
                """\
                from pandas import Series
                def my_custom_ind(close, length=10, offset=None, **kwargs):
                    return close.rolling(length).mean()
                def my_custom_ind_method(self, length=10, offset=None, **kwargs):
                    return my_custom_ind(
                        self._get_column(self._data, 'close'),
                        length=length, **kwargs
                    )
            """
            )
            with open(os.path.join(cat_dir, "my_custom_ind.py"), "w") as f:
                f.write(indicator_src)
            custom.import_dir(tmp, verbose=False)
            self.assertTrue(hasattr(pandas_ta_classic, "my_custom_ind"))
