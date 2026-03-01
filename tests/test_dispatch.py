"""Generic dispatch tests: ensure all mamode/ma variants produce valid results.

This catches bugs like the zlma import issue where non-default mamode branches
were silently broken because submodule objects were imported instead of functions.
"""

from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame, Series


class TestMamodeDispatch(TestCase):
    """Test that every indicator with mamode/ma dispatch actually works
    for all documented modes."""

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    # -- Indicators using ma() dispatcher (safe, but verify) --

    def test_atr_mamodes(self):
        for mode in ["ema", "sma", "rma", "dema", "hma"]:
            result = pandas_ta.atr(
                self.high, self.low, self.close, mamode=mode, talib=False
            )
            self.assertIsInstance(result, Series, msg=f"atr(mamode={mode!r})")

    def test_bbands_mamodes(self):
        for mode in ["ema", "sma", "rma"]:
            result = pandas_ta.bbands(self.close, mamode=mode, talib=False)
            self.assertIsInstance(result, DataFrame, msg=f"bbands(mamode={mode!r})")

    def test_kc_mamodes(self):
        for mode in ["ema", "sma"]:
            result = pandas_ta.kc(self.high, self.low, self.close, mamode=mode)
            self.assertIsInstance(result, DataFrame, msg=f"kc(mamode={mode!r})")

    def test_macd_mamodes(self):
        for mode in ["ema", "sma"]:
            result = pandas_ta.macd(self.close, mamode=mode, talib=False)
            self.assertIsInstance(result, DataFrame, msg=f"macd(mamode={mode!r})")

    def test_apo_mamodes(self):
        for mode in ["ema", "sma"]:
            result = pandas_ta.apo(self.close, mamode=mode, talib=False)
            self.assertIsInstance(result, Series, msg=f"apo(mamode={mode!r})")

    def test_ppo_mamodes(self):
        for mode in ["ema", "sma"]:
            result = pandas_ta.ppo(self.close, mamode=mode, talib=False)
            self.assertIsInstance(result, DataFrame, msg=f"ppo(mamode={mode!r})")

    def test_stoch_mamodes(self):
        for mode in ["ema", "sma"]:
            result = pandas_ta.stoch(
                self.high, self.low, self.close, mamode=mode, talib=False
            )
            self.assertIsInstance(result, DataFrame, msg=f"stoch(mamode={mode!r})")

    def test_bias_mamodes(self):
        for mode in ["ema", "sma", "rma"]:
            result = pandas_ta.bias(self.close, mamode=mode)
            self.assertIsInstance(result, Series, msg=f"bias(mamode={mode!r})")

    # -- Indicators with if/elif dispatch (higher risk of import bugs) --

    def test_zlma_mamodes(self):
        modes = [
            "dema",
            "ema",
            "hma",
            "linreg",
            "rma",
            "sma",
            "swma",
            "t3",
            "tema",
            "trima",
            "vidya",
            "wma",
        ]
        for mode in modes:
            result = pandas_ta.zlma(self.close, mamode=mode)
            self.assertIsInstance(result, Series, msg=f"zlma(mamode={mode!r})")

    def test_qstick_mamodes(self):
        for mode in ["dema", "ema", "hma", "rma", "sma"]:
            result = pandas_ta.qstick(self.open, self.close, ma=mode)
            self.assertIsInstance(result, Series, msg=f"qstick(ma={mode!r})")

    # -- ma() dispatcher itself --

    def test_ma_all_modes(self):
        from pandas_ta_classic.overlap.ma import _MA_DISPATCH

        for mode in _MA_DISPATCH:
            if mode == "mama":
                continue  # mama returns (Series, Series) tuple, skip
            result = pandas_ta.ma(mode, self.close, length=10)
            self.assertIsInstance(result, Series, msg=f"ma({mode!r})")

    def test_ma_no_args(self):
        result = pandas_ta.ma(None, None)
        self.assertIsInstance(result, list)

    def test_ma_invalid_mode(self):
        result = pandas_ta.ma("nonexistent", self.close, length=10)
        self.assertIsInstance(result, Series)
