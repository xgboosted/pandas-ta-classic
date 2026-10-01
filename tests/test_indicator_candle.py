from unittest import TestCase, skipUnless

import numpy as np
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.assertions import IndicatorSpec, assert_indicator_standard, assert_talib
from tests.config import get_sample_data

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TestCandle(TestCase):
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

    def test_ha(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ha,
                args=[self.open, self.high, self.low, self.close],
                expected_name="Heikin-Ashi",
                expected_type=DataFrame,
                expected_columns=["HA_open", "HA_high", "HA_low", "HA_close"],
                none_arg_idx=0,
            ),
        )

    def test_cdl_pattern(self):
        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="all")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(len(result.columns), len(pandas_ta.ALL_PATTERNS))

        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="doji")
        self.assertIsInstance(result, DataFrame)

        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name=["doji", "inside"])
        self.assertIsInstance(result, DataFrame)

    def test_cdl_doji(self):
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(
                self,
                result,
                talib.CDLDOJI(self.open, self.high, self.low, self.close),
                correlation_threshold=0.99,
            )
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_doji,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_DOJI_10_0.1",
                expected_type=Series,
                none_arg_idx=0,
            ),
        )

    def test_cdl_inside(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_inside,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_INSIDE",
                expected_type=Series,
                none_arg_idx=0,
            ),
        )

        result = pandas_ta.cdl_inside(self.open, self.high, self.low, self.close, asbool=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CDL_INSIDE")

    def test_cdl_z(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cdl_z,
                args=[self.open, self.high, self.low, self.close],
                expected_name="CDL_Z_30_1",
                expected_type=DataFrame,
                expected_columns=[
                    "open_Z_30_1",
                    "high_Z_30_1",
                    "low_Z_30_1",
                    "close_Z_30_1",
                ],
                none_arg_idx=0,
            ),
        )


# Sixteen calm bars: real body 0.5, high-low range 10. They set the rolling
# averages the pattern conditions compare against -- BodyLong and BodyShort at
# 0.5, BodyDoji and ShadowVeryShort at 0.1 * 10 = 1.0 -- so the pattern bars
# below are long, short or shadowless against a known threshold.
_CALM_BARS = 16


def _pattern_frame(bars: list[tuple[float, float, float, float]]) -> dict[str, Series]:
    """The calm run followed by *bars*, as ``(open, high, low, close)`` Series."""
    rows = [(100.0 if i % 2 == 0 else 100.5, 105.0, 95.0, 100.5 if i % 2 == 0 else 100.0) for i in range(_CALM_BARS)]
    rows.extend(bars)
    columns = list(zip(*rows, strict=True))
    return {name: Series(values, dtype=float) for name, values in zip(("open", "high", "low", "close"), columns, strict=True)}


# Fourteen patterns whose signal branch the oracle suite never ran: eight do not
# occur once in the 5241 bars of SPY_D, so the comparison against TA-Lib only ever
# proved that both sides agree on "no signal", and six fire between one and four
# times, resting it on a handful of rows. Each entry is the shortest run of bars
# that satisfies the pattern's conditions, built from its own condition block (a
# random search supplied the starting point for some of the eight) and reduced by
# hand to round numbers; the signal lands on the last bar. Frozen literals --
# nothing is searched at test time.
# (name, TA-Lib name, kwargs, bars as (open, high, low, close), expected signal)
_PATTERN_CASES = [
    # long black, long lower shadow, no upper shadow / smaller black opening inside the 1st, holding above its low / small black marubozu
    #   engulfed by the 2nd
    ("cdl_3starsinsouth", "CDL3STARSINSOUTH", {}, [(120, 120, 95, 110), (115, 116, 100, 108), (105.3, 105.4, 104.9, 105)], 100),
    # long black / doji gapping down, shadows clear of the 1st / long white gapping back up over the doji
    ("cdl_abandonedbaby", "CDLABANDONEDBABY", {"penetration": 0.3}, [(120, 120, 110, 110), (105, 105.2, 104.8, 105), (106, 118, 106, 118)], 100),
    # long white / white gapping up / higher high and low / higher again / black closing inside the gap
    (
        "cdl_breakaway",
        "CDLBREAKAWAY",
        {},
        [(110, 120, 110, 120), (125, 128, 124, 127), (127, 130, 126, 129), (129, 133, 128, 132), (132, 133, 121, 122)],
        -100,
    ),
    # black marubozu / black marubozu / black gapping down, upper shadow back into the 2nd body / black engulfing the 3rd, shadows included
    ("cdl_concealbabyswall", "CDLCONCEALBABYSWALL", {}, [(120, 120, 110, 110), (110, 110, 100, 100), (95, 101, 89, 90), (99, 102, 85, 88)], 100),
    # white marubozu / black marubozu gapping down below it
    ("cdl_kicking", "CDLKICKING", {}, [(100, 110, 100, 110), (97, 97, 85, 85)], -100),
    # white marubozu, body 10 / black marubozu, body 12 -- the longer one names the signal
    ("cdl_kickingbylength", "CDLKICKINGBYLENGTH", {}, [(100, 110, 100, 110), (97, 97, 85, 85)], -100),
    # long white / small black gapping up / reaction day holding above half the 1st body / lower again, still holding / white closing above
    #   every reaction high
    (
        "cdl_mathold",
        "CDLMATHOLD",
        {"penetration": 0.5},
        [(110, 121, 110, 120), (122, 122.5, 121.5, 121.7), (119.3, 119.6, 118.8, 119), (118.6, 119, 118, 118.4), (119, 124, 119, 124)],
        100,
    ),
    # long white / three small black bars falling inside its range / long white closing above the 1st close
    (
        "cdl_risefall3methods",
        "CDLRISEFALL3METHODS",
        {},
        [(110, 121, 109, 120), (119.8, 120, 119.3, 119.5), (119.3, 119.5, 118.8, 119), (118.8, 119, 118.3, 118.5), (119, 131, 119, 131)],
        100,
    ),
    # long white marubozu / black gapping up over the 1st body / black opening inside the 2nd body, closing inside the 1st
    ("cdl_2crows", "CDL2CROWS", {}, [(100, 110, 100, 110), (118, 119, 112, 112), (116, 116.5, 104, 105)], -100),
    # prior white, high above the 1st crow's close / 1st crow: black, no lower shadow / 2nd crow opens inside the 1st body / 3rd crow opens
    #   inside the 2nd body, closes lowest
    ("cdl_3blackcrows", "CDL3BLACKCROWS", {}, [(100, 120, 100, 120), (118, 118, 112, 112), (116, 116, 106, 106), (110, 110, 100, 100)], -100),
    # 1st soldier: white marubozu, no upper shadow / 2nd opens inside the 1st body, body not far shorter / 3rd opens inside the 2nd body,
    #   highest close
    ("cdl_3whitesoldiers", "CDL3WHITESOLDIERS", {}, [(100, 110, 100, 110), (105, 118, 105, 118), (115, 128, 115, 128)], 100),
    # 1st crow: black marubozu / 2nd opens exactly at the 1st close -- "identical" / 3rd opens exactly at the 2nd close, closes lowest
    ("cdl_identical3crows", "CDLIDENTICAL3CROWS", {}, [(120, 120, 110, 110), (110, 110, 100, 100), (100, 100, 90, 90)], -100),
    # 1st white / 2nd white opens inside the 1st body, closes higher / 3rd white opens inside the 2nd body, closes higher / black engulfing
    #   all three: opens above, closes below
    ("cdl_3linestrike", "CDL3LINESTRIKE", {}, [(100, 110, 100, 110), (105, 115, 105, 115), (110, 120, 110, 120), (122, 122, 95, 95)], 100),
    # black; its close is the sandwich level / white trading entirely above that close / black closing back at the level
    ("cdl_sticksandwich", "CDLSTICKSANDWICH", {}, [(110, 110, 100, 100), (105, 115, 105, 115), (112, 112, 99, 100)], 100),
]


class TestCandlePatternsOnBarsThatTriggerThem(TestCase):
    """Fourteen patterns on bars built to satisfy their conditions.

    `tests/test_oracle_talib.py` compares all 62 patterns against TA-Lib over
    SPY_D and they agree on every bar -- but eight of them return 0 for all 5241
    rows, so that agreement said nothing about the branch that emits the signal,
    and six more fire between one and four times, resting the comparison on a
    handful of rows. The bars are frozen literals, not a search at test time:
    each run was built from the pattern's own condition block (a random search
    supplied the starting point for some of the eight) and then reduced to round
    numbers that still satisfy every clause.
    """

    def _frame(self, bars):
        frame = _pattern_frame(bars)
        return frame["open"], frame["high"], frame["low"], frame["close"]

    def _native(self, name):
        # `ta.cdl_kicking` is the submodule; the callable of the same name lives
        # inside it, as `tests/test_oracle_talib.py` resolves it too.
        return getattr(getattr(pandas_ta, name), name)

    def test_each_pattern_fires_on_its_own_bars(self):
        """The native implementation reports the signal on the last bar."""
        for name, _talib_name, kwargs, bars, expected in _PATTERN_CASES:
            with self.subTest(pattern=name):
                result = self._native(name)(*self._frame(bars), **kwargs)
                self.assertIsInstance(result, Series)
                self.assertEqual(int(result.iloc[-1]), expected)
                # Nothing else in the frame may fire, or the case would not pin
                # down which bars produced the signal.
                self.assertEqual(int(result.iloc[:-1].abs().sum()), 0)

    @skipUnless(HAS_TALIB, "TA-Lib is not installed")
    def test_each_pattern_matches_talib_on_its_own_bars(self):
        """Bar for bar, including the signal itself.

        `talib=True` falls back to the native formula when TA-Lib is missing, so
        this comparison is worthless without it -- hence the skip.
        """
        for name, talib_name, kwargs, bars, _expected in _PATTERN_CASES:
            with self.subTest(pattern=name):
                open_, high, low, close = self._frame(bars)
                native = self._native(name)(open_, high, low, close, **kwargs)
                expected = getattr(talib, talib_name)(open_, high, low, close, **kwargs)
                np.testing.assert_array_equal(native.to_numpy(dtype=float), np.asarray(expected, dtype=float), err_msg=f"{name} differs from TA-Lib")
