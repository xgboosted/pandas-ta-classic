from unittest import TestCase, skipUnless

import numpy as np
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.candles._cdl_math import AVG_FACTOR, CandleArrays, CandleSetting, candle_average, candle_avg_period
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

        # An empty selection has nothing to build a frame from.
        self.assertIsNone(pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name=[]))

    def test_every_listed_pattern_has_an_implementation(self):
        """ALL_PATTERNS is hand-maintained; the modules are discovered.

        cdl_pattern() rejects a name in neither list, so a drift between the two
        would leave a listed pattern with no branch to run. The TA-Lib fallback
        that used to absorb that was dead code: every listed pattern is native.
        """
        from pandas_ta_classic.candles.cdl_pattern import (
            _NATIVE_PATTERNS,
            ALL_PATTERNS,
            cdl_doji,
            cdl_inside,
        )

        implemented = set(_NATIVE_PATTERNS) | {"doji", "inside"}
        self.assertEqual(set(ALL_PATTERNS), implemented)
        self.assertEqual(len(ALL_PATTERNS), len(set(ALL_PATTERNS)))
        self.assertTrue(callable(cdl_doji) and callable(cdl_inside))

    def test_cdl_doji(self):
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close, talib=False)
        if HAS_TALIB:
            assert_talib(self, result, talib.CDLDOJI(self.open, self.high, self.low, self.close))
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

    def test_cdl_z_ddof_changes_the_values(self):
        """ddof was validated and named the columns but had no numeric effect.

        zscore() had no such parameter, so the forwarded value landed in
        **kwargs and was dropped; ddof=0 and ddof=1 were bit-identical.
        """
        args = (self.open, self.high, self.low, self.close)
        sample = pandas_ta.cdl_z(*args, ddof=1)
        population = pandas_ta.cdl_z(*args, ddof=0)

        self.assertEqual(list(population.columns), ["open_Z_30_0", "high_Z_30_0", "low_Z_30_0", "close_Z_30_0"])
        self.assertGreater(float(np.nanmax(np.abs(population.to_numpy() - sample.to_numpy()))), 0.0)
        # The default is still the sample deviation.
        np.testing.assert_array_equal(pandas_ta.cdl_z(*args).to_numpy(), sample.to_numpy())

        # full=True is anchored and takes no ddof, so it keeps its own suffix.
        self.assertEqual(list(pandas_ta.cdl_z(*args, full=True).columns), ["open_Za", "high_Za", "low_Za", "close_Za"])


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
    # white marubozu, body 12 / black marubozu, body 10 -- the *first* marubozu is
    # longer, so the signal is the opposite of cdl_kicking's: that is what tells
    # the two apart, and the shared bars of cdl_kicking would not.
    ("cdl_kickingbylength", "CDLKICKINGBYLENGTH", {}, [(100, 112, 100, 112), (97, 97, 87, 87)], 100),
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
    # The six two-sided patterns above are mirrored -- p -> 200 - p, high and
    # low swapped -- so the opposite-sign branch fires too. Each is the same
    # case upside down, and native matches TA-Lib on every bar.
    # long white / doji gapping up, shadows clear of the 1st / long black gapping back down over the doji
    ("cdl_abandonedbaby", "CDLABANDONEDBABY", {"penetration": 0.3}, [(80, 90, 80, 90), (95, 95.2, 94.8, 95), (94, 94, 82, 82)], -100),
    # long black / black gapping down / lower high and low / lower again / white closing inside the gap
    (
        "cdl_breakaway",
        "CDLBREAKAWAY",
        {},
        [(90, 90, 80, 80), (75, 76, 72, 73), (73, 74, 70, 71), (71, 72, 67, 68), (68, 79, 67, 78)],
        100,
    ),
    # black marubozu / white marubozu gapping up above it
    ("cdl_kicking", "CDLKICKING", {}, [(100, 100, 90, 90), (103, 115, 103, 115)], 100),
    # black marubozu, body 12 / white marubozu, body 10 -- the *first* is longer
    ("cdl_kickingbylength", "CDLKICKINGBYLENGTH", {}, [(100, 100, 88, 88), (103, 113, 103, 113)], -100),
    # long black / three small white bars rising inside its range / long black closing below the 1st close
    (
        "cdl_risefall3methods",
        "CDLRISEFALL3METHODS",
        {},
        [(90, 91, 79, 80), (80.2, 80.7, 80, 80.5), (80.7, 81.2, 80.5, 81), (81.2, 81.7, 81, 81.5), (81, 81, 69, 69)],
        -100,
    ),
    # 1st black / 2nd black opens inside the 1st body, closes lower / 3rd black opens inside the 2nd body, closes lower / white engulfing
    #   all three: opens below, closes above
    ("cdl_3linestrike", "CDL3LINESTRIKE", {}, [(100, 100, 90, 90), (95, 95, 85, 85), (90, 90, 80, 80), (78, 105, 78, 105)], -100),
]


class TestCandlePatternsOnBarsThatTriggerThem(TestCase):
    """Fourteen patterns on bars built to satisfy their conditions, plus the six
    two-sided ones mirrored to fire the opposite sign.

    `tests/test_oracle_talib.py` compares 59 patterns against TA-Lib over the
    last 2000 bars of SPY_D and they agree on every bar -- but eight of them
    return 0 for all 2000 rows, so that agreement said nothing about the branch
    that emits the signal, and six more fire between one and four times, resting
    the comparison on a handful of rows. The bars are frozen literals, not a
    search at test time: each run was built from the pattern's own condition
    block (a random search supplied the starting point for some of the eight)
    and then reduced to round numbers that still satisfy every clause.
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

        The comparison calls `talib.<name>` directly, which needs TA-Lib
        installed; the `cdl_*` functions have no `talib` parameter to fall back.
        """
        for name, talib_name, kwargs, bars, _expected in _PATTERN_CASES:
            with self.subTest(pattern=name):
                open_, high, low, close = self._frame(bars)
                native = self._native(name)(open_, high, low, close, **kwargs)
                expected = getattr(talib, talib_name)(open_, high, low, close, **kwargs)
                np.testing.assert_array_equal(native.to_numpy(dtype=float), np.asarray(expected, dtype=float), err_msg=f"{name} differs from TA-Lib")


class TestCandleAverage(TestCase):
    """`candle_average` against the per-pattern bookkeeping it replaced.

    Every pattern used to carry TA-Lib's ``PeriodTotal`` as a scalar: seed it,
    compare ``factor * total``, then ``total += range[i - lag] - range[trail - lag]``.
    The helper must reproduce that bit for bit, or a threshold that lands
    exactly on a candle's range would flip a signal.
    """

    @classmethod
    def setUpClass(cls):
        # Bodies and shadows spread over many orders of magnitude, with opens
        # near zero so the ranges keep their full mantissa: their window sums
        # then round differently depending on the order they are added in. On
        # price-like bars the ranges sit on one grid, both seeds add up exactly
        # and the test could not tell them apart.
        rng = np.random.default_rng(7)
        n = 3000
        open_ = rng.normal(0, 1e-3, n)
        close = open_ + rng.lognormal(0, 3, n) * rng.choice([-1.0, 1.0], n)
        high = np.maximum(open_, close) + rng.lognormal(0, 3, n)
        low = np.minimum(open_, close) - rng.lognormal(0, 3, n)
        cls.ca = CandleArrays(open_, high, low, close)

    @staticmethod
    def _scalar_bookkeeping(arr, period, lag, start_idx, factor, sequential_seed):
        out = np.full(len(arr), np.nan)
        window = arr[start_idx - lag - period : start_idx - lag]
        if sequential_seed:
            total = 0.0
            for value in window:
                total += value
        else:
            total = float(window.sum())
        trail = start_idx - period
        for i in range(start_idx, len(arr)):
            out[i] = factor * (arr[i - lag] if period == 0 else total)
            total += arr[i - lag] - arr[trail - lag]
            trail += 1
        return out

    def test_bit_identical_to_scalar_bookkeeping(self):
        for setting in CandleSetting:
            period = candle_avg_period(setting)
            for lag in range(5):
                for extra in (0, 3, 7):
                    for sequential_seed in (False, True):
                        with self.subTest(setting=setting.name, lag=lag, extra=extra, sequential_seed=sequential_seed):
                            start_idx = period + lag + extra
                            got = candle_average(self.ca, setting, lag, start_idx, sequential_seed=sequential_seed)
                            want = self._scalar_bookkeeping(self.ca._ranges[setting], period, lag, start_idx, AVG_FACTOR[setting], sequential_seed)
                            self.assertTrue(np.isnan(got[:start_idx]).all())
                            np.testing.assert_array_equal(got[start_idx:].view(np.int64), want[start_idx:].view(np.int64))

    def test_the_two_seeds_differ_on_this_data(self):
        """Guards the test above: if both seeds agreed here, it would not notice a helper that ignores the flag."""
        differing = 0
        for setting in CandleSetting:
            start_idx = candle_avg_period(setting) + 4
            for lag in range(5):
                pairwise = candle_average(self.ca, setting, lag, start_idx)
                sequential = candle_average(self.ca, setting, lag, start_idx, sequential_seed=True)
                differing += int((pairwise[start_idx:].view(np.int64) != sequential[start_idx:].view(np.int64)).any())
        self.assertGreater(differing, 0)

    def test_start_before_the_first_window_raises(self):
        period = candle_avg_period(CandleSetting.BodyLong)
        with self.assertRaisesRegex(ValueError, r"candle_average\(\) start_idx must be >= lag \+ period"):
            candle_average(self.ca, CandleSetting.BodyLong, 2, period + 1)
        with self.assertRaisesRegex(ValueError, "start_idx"):
            candle_average(self.ca, CandleSetting.ShadowLong, 3, 2)
