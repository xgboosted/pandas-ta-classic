from unittest import TestCase

import numpy as np
from pandas import Series

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestMath(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    # -----------------------------------------------------------------------
    # Rolling operators
    # -----------------------------------------------------------------------

    def test_rolling_max_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_max(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_max(self.close, length=-5)

    def test_rolling_min_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_min(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_min(self.close, length=-5)

    def test_rolling_sum_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_sum(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_sum(self.close, length=-5)

    def test_window_index_convention(self):
        # position counted from the oldest bar of the window, not "bars back"; the oldest wins a tie
        s = Series([1.0, 5.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(pandas_ta.maxindex(s, length=5).tolist()[4:], [1.0, 0.0])
        self.assertEqual(pandas_ta.minindex(s, length=5).tolist()[4:], [0.0, 1.0])
        self.assertEqual(pandas_ta.minmaxindex(s, length=5).iloc[4:].to_numpy().tolist(), [[0.0, 1.0], [1.0, 0.0]])
        # the same tie rule on the minimum side: two lows of 1, at positions 1 and 3
        lows = Series([5.0, 1.0, 3.0, 1.0, 4.0])
        self.assertEqual(pandas_ta.minindex(lows, length=5).iloc[-1], 1.0)
        self.assertEqual(pandas_ta.minmaxindex(lows, length=5).iloc[-1].tolist(), [1.0, 0.0])

    def test_window_index_window_with_a_missing_bar_is_nan(self):
        # argmax would point at the NaN; a window holding one has no position to report
        s = Series([1.0, 5.0, np.nan, 3.0, 4.0, 2.0, 6.0])
        for func in (pandas_ta.maxindex, pandas_ta.minindex):
            self.assertEqual(np.isnan(func(s, length=3).to_numpy()).tolist(), [True] * 5 + [False] * 2)
