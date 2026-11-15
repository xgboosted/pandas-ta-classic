from unittest import TestCase

from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestVolatility(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cvi(self):
        self.assertIsNone(pandas_ta.cvi(self.high, None))

    def test_bbands(self):
        result = pandas_ta.bbands(self.close, talib=False)

        result = pandas_ta.bbands(self.close, ddof=0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        # A non-default ddof changes the values, so it is in the name (#343)
        result = pandas_ta.bbands(self.close, ddof=1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0_ddof1")

    def test_donchian(self):
        result = pandas_ta.donchian(self.high, self.low, lower_length=20, upper_length=5)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_5")

    def test_kc(self):
        result = pandas_ta.kc(self.high, self.low, self.close, mamode="sma")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCs_20_2")

        result = pandas_ta.kc(self.high, self.low, self.close, tr=False)
        self.assertIsInstance(result, DataFrame)

    def test_rvi(self):
        result = pandas_ta.rvi(self.close, self.high, self.low, refined=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIr_14")

        result = pandas_ta.rvi(self.close, self.high, self.low, thirds=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RVIt_14")

    def test_rvi_mode_arguments_fail_fast(self):
        # refined/thirds used to be bool()-coerced, so rvi(refined="yes") ran
        # the refined branch without a word.
        for kwarg in ("refined", "thirds"):
            with self.subTest(kwarg=kwarg), self.assertRaisesRegex(ValueError, rf"rvi\(\) {kwarg} must be True or False"):
                pandas_ta.rvi(self.close, self.high, self.low, **{kwarg: "yes"})

        # They are alternative modes; refined used to win silently.
        with self.assertRaisesRegex(ValueError, "alternative modes"):
            pandas_ta.rvi(self.close, self.high, self.low, refined=True, thirds=True)

        # Both modes are defined over high/low. Omitting them used to return
        # None, which reads as an ordinary short-input result.
        for kwarg in ("refined", "thirds"):
            with self.subTest(kwarg=kwarg), self.assertRaisesRegex(ValueError, rf"rvi\(\) {kwarg}=True needs both high and low"):
                pandas_ta.rvi(self.close, **{kwarg: True})
        with self.assertRaisesRegex(ValueError, "needs both high and low"):
            pandas_ta.rvi(self.close, high=self.high, refined=True)

    def test_ui(self):
        result = pandas_ta.ui(self.close, everget=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "UIe_14")

    def test_hwc(self):
        result = pandas_ta.hwc(self.close, channel_eval=True)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("HWW", result.columns)
        self.assertIn("HWPCT", result.columns)
