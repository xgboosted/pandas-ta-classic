from tests.config import get_sample_data

from unittest import TestCase
from pandas import DataFrame, Series


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

    def test_all_indicators_return_series_or_dataframe(self):
        indicator_names = self.data.ta.indicators(as_list=True)
        failures = []

        for name in indicator_names:
            try:
                result = getattr(self.data.ta, name)()
            except Exception:
                continue

            if not isinstance(result, (Series, DataFrame)):
                failures.append(f"{name}: got {type(result).__name__}")

        self.assertEqual(failures, [], f"Indicators returning wrong type: {failures}")
