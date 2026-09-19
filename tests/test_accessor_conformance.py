from unittest import TestCase

from pandas import DataFrame, Series

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.config import get_sample_data


class TestAccessorConformance(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Called with no arguments these cannot produce a column: they need an input
    # the DataFrame does not hold (a benchmark, a pair of Series, a trend, an MA
    # name), so df.ta.<name>() returns None.  They used to return the source
    # DataFrame, which made "nothing was produced" indistinguishable from a
    # result without comparing object identity.
    NEEDS_AN_EXTRA_ARGUMENT = frozenset({"beta", "correl", "long_run", "ma", "short_run", "tsignals", "xsignals"})

    def test_all_indicators_return_series_or_dataframe(self):
        indicator_names = self.data.ta.indicators(as_list=True)
        failures = []

        for name in indicator_names:
            try:
                result = getattr(self.data.ta, name)()
            except Exception:  # noqa: BLE001, S112 - indicators that need extra inputs are out of scope here
                continue

            if result is None and name in self.NEEDS_AN_EXTRA_ARGUMENT:
                continue
            if not isinstance(result, (Series, DataFrame)):
                failures.append(f"{name}: got {type(result).__name__}")

        self.assertEqual(failures, [], f"Indicators returning wrong type: {failures}")

    def test_indicators_without_their_required_input_return_none(self):
        """Not the source DataFrame, which silently looked like a successful run."""
        for name in sorted(self.NEEDS_AN_EXTRA_ARGUMENT):
            with self.subTest(indicator=name):
                self.assertIsNone(getattr(self.data.ta, name)())
