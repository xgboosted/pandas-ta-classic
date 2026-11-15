"""Tests of pandas_ta_classic/utils/_core.py: series checks, offset and fill, parameter validation."""

import inspect
import math
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, strategies as st
from pandas import DataFrame, Series

import pandas_ta_classic as ta
from pandas_ta_classic.utils import apply_fill, apply_offset, get_drift, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_float, _pos_int, _str_param
from tests.config import get_sample_data
from tests.strategies import price_series


@pytest.mark.parametrize("bad", [0, -5, 2.7, True, np.nan, float("inf"), "14"])
def test_pos_int_rejects(bad):
    def sma(length=None):
        return _pos_int(length, 10, "length")

    with pytest.raises(ValueError, match=r"sma\(\) length must be an integer > 0"):
        sma(bad)


def test_pos_helpers_accept_none_and_valid_values():
    assert _pos_int(None, 10, "length") == 10
    assert _pos_int(14, 10, "length") == 14
    assert _pos_int(14.0, 10, "length") == 14
    assert _pos_int(np.int64(5), 10, "length") == 5
    assert _pos_float(0.5, 0.2, "na", lt=1) == 0.5
    assert _pos_int(0, 1, "ddof", gt=None, ge=0, lt=30) == 0


def test_a_whole_number_keeps_the_type_of_an_int_default():
    assert type(_pos_float(2.0, 2, "scalar")) is int
    assert type(_pos_float(2, 2.0, "std")) is float
    assert type(_pos_float(2.5, 2, "scalar")) is float
    assert type(_number(100.0, 100, "scalar")) is int


def test_number_accepts_zero_and_negatives_but_not_junk():
    assert _number(None, 100, "scalar") == 100
    assert _number(0, 100, "scalar") == 0.0  # used to become 100
    assert _number(-2, 100, "scalar") == -2.0
    for bad in (np.nan, float("inf"), True, "100"):
        with pytest.raises(ValueError, match="scalar must be a number"):
            _number(bad, 100, "scalar")


def test_bool_param_rejects_non_bools():
    def ema(talib=None):
        return _bool_param(talib, False, "talib")

    assert ema() is False
    assert ema(True) is True
    assert ema(np.bool_(True)) is True
    for bad in (1, 0, "True", []):  # talib=1 used to mean False
        with pytest.raises(ValueError, match=r"ema\(\) talib must be True or False"):
            ema(bad)


def test_str_param_rejects_non_strings_and_unknown_choices():
    def cpr(method=None):
        return _str_param(method, "classic", "method", choices={"classic", "woodie"})

    assert cpr() == "classic"
    assert cpr("WOODIE") == "woodie"
    for bad in (1, "", "fibonaci"):
        with pytest.raises(ValueError, match=r"cpr\(\) method must be"):
            cpr(bad)


def test_bad_fill_method_leaves_the_input_unfilled():
    """fill_method is validated before fillna runs: a rejected call must not have
    filled the caller's Series in place on its way to the error (AGENTS rule 19)."""
    series = Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="fill_method"):
        ta.utils.apply_fill(series, fillna=0, fill_method="bogus")
    assert series.isna().tolist() == [False, True, False]


class TestCoreUtilities(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        self.utils = ta.utils

    def tearDown(self):
        del self.utils

    def test__add_prefix_suffix(self):
        result = self.data.ta.hl2(append=False, prefix="pre")
        self.assertEqual(result.name, "pre_HL2")

        result = self.data.ta.hl2(append=False, suffix="suf")
        self.assertEqual(result.name, "HL2_suf")

        result = self.data.ta.hl2(append=False, prefix="pre", suffix="suf")
        self.assertEqual(result.name, "pre_HL2_suf")

        result = self.data.ta.hl2(append=False, prefix=1, suffix=2)
        self.assertEqual(result.name, "1_HL2_2")

        result = self.data.ta.macd(append=False, prefix="pre", suffix="suf")
        for col in result.columns:
            self.assertTrue(col.startswith("pre_") and col.endswith("_suf"))

    def test_recent_extreme_index(self):
        # Both count backwards from the last bar, so 0 means "the last bar".
        rising = Series([1.0, 2.0, 3.0])
        self.assertEqual(self.utils.recent_maximum_index(rising), 0)
        self.assertEqual(self.utils.recent_minimum_index(rising), 2)

        falling = Series([3.0, 2.0, 1.0])
        self.assertEqual(self.utils.recent_maximum_index(falling), 2)
        self.assertEqual(self.utils.recent_minimum_index(falling), 0)

        # Ties resolve to the most recent bar.
        self.assertEqual(self.utils.recent_maximum_index(Series([5.0, 1.0, 5.0])), 0)

    def test_signed_series(self):
        series = Series([3.0, 2, 2, 1, 1, 5, 6, 6, 7, 5])
        result = self.utils.signed_series(series)
        self.assertIsInstance(result, Series)
        self.assertTrue(np.isnan(result.iloc[0]))
        np.testing.assert_array_equal(
            result.iloc[1:].to_numpy(),
            np.array([-1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0]),
        )

        # initial replaces the undefined first difference.
        self.assertEqual(self.utils.signed_series(series, initial=1).iloc[0], 1)

        # A non-number used to fail inside pandas as "Invalid value 'zz' for
        # dtype 'float64'", naming neither the function nor the parameter.
        for bad in ("zz", True, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "signed_series\\(\\) initial must be a number"):
                self.utils.signed_series(series, initial=bad)

    def test_is_percent(self):
        for value in (0, 50, 100, 99.5):
            with self.subTest(value=value):
                self.assertTrue(self.utils.is_percent(value))
        for value in (-1, 100.1, 101):
            with self.subTest(value=value):
                self.assertFalse(self.utils.is_percent(value))
        # A non-number is not a percent rather than a TypeError.
        self.assertFalse(self.utils.is_percent("50"))
        self.assertFalse(self.utils.is_percent(None))
        # bool is an int subclass: True used to read as 1, a percent.
        self.assertFalse(self.utils.is_percent(True))
        self.assertFalse(self.utils.is_percent(False))

    def test_unsigned_differences_default(self):
        # the docstring's example; without asint the flags stay float, even for an int series
        series = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5, 3])
        positive, negative = self.utils.unsigned_differences(series)
        self.assertEqual(positive.dtype, np.float64)
        self.assertEqual(negative.dtype, np.float64)
        np.testing.assert_array_equal(positive.to_numpy(), [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(negative.to_numpy(), [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
        # None selects the default, like every other option
        for flags, default in zip(self.utils.unsigned_differences(series, asint=None), (positive, negative)):
            pd.testing.assert_series_equal(flags, default)

    def test_unsigned_differences_asint(self):
        series = Series([1.0, 2.0, 2.0, 1.0])
        positive, negative = self.utils.unsigned_differences(series, asint=True)
        self.assertEqual(positive.dtype.kind, "i")
        self.assertEqual(negative.dtype.kind, "i")
        np.testing.assert_array_equal(positive.to_numpy(), np.array([0, 1, 0, 0]))
        np.testing.assert_array_equal(negative.to_numpy(), np.array([0, 0, 0, 1]))

    def test_tal_ma(self):
        self.assertEqual(self.utils.tal_ma("sma"), 0)
        self.assertEqual(self.utils.tal_ma("Sma"), 0)
        self.assertEqual(self.utils.tal_ma("ema"), 1)
        self.assertEqual(self.utils.tal_ma("wma"), 2)
        self.assertEqual(self.utils.tal_ma("dema"), 3)
        self.assertEqual(self.utils.tal_ma("tema"), 4)
        self.assertEqual(self.utils.tal_ma("trima"), 5)
        self.assertEqual(self.utils.tal_ma("kama"), 6)
        self.assertEqual(self.utils.tal_ma("mama"), 7)
        self.assertEqual(self.utils.tal_ma("t3"), 8)

        # An unknown name or a non-str raises instead of silently returning SMA.
        with self.assertRaisesRegex(ValueError, "Unknown TA-Lib MA type"):
            self.utils.tal_ma("nope")
        with self.assertRaisesRegex(TypeError, "expects a str MA name"):
            self.utils.tal_ma(5)

    def test_get_drift(self):
        self.assertEqual(self.utils.get_drift(None), 1)
        self.assertEqual(self.utils.get_drift(3), 3)
        # 0, negatives, fractions, bools and non-numbers used to become 1 silently
        for bad in (0, -1, 1.1, True, "", [], {}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "drift must be an integer > 0"):
                self.utils.get_drift(bad)

    def test_get_offset(self):
        self.assertEqual(self.utils.get_offset(None), 0)
        self.assertEqual(self.utils.get_offset(0), 0)
        self.assertEqual(self.utils.get_offset(1), 1)
        self.assertEqual(self.utils.get_offset(-2), -2)
        # fractions, bools and non-numbers used to become 0 silently
        for bad in (-1.1, 2.5, True, "", [], {}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "offset must be an integer"):
                self.utils.get_offset(bad)

    def test_version(self):
        result = ta.version
        self.assertIsInstance(result, str)


class TestApplyOffset(TestCase):
    """Unit tests for apply_offset (utils/_core.py)."""

    def setUp(self):
        self.s = Series([1.0, 2.0, 3.0, 4.0, 5.0], name="test")
        self.df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    # -- no-op cases --

    def test_offset_zero_returns_same_object(self):
        result = ta.utils.apply_offset(self.s, 0)
        self.assertIs(result, self.s)

    def test_offset_default_zero(self):
        result = ta.utils.apply_offset(self.s)
        self.assertIs(result, self.s)

    # -- single Series --

    def test_positive_offset(self):
        result = ta.utils.apply_offset(self.s, 2)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertEqual(result.iloc[2], 1.0)
        self.assertEqual(result.iloc[4], 3.0)

    def test_negative_offset(self):
        result = ta.utils.apply_offset(self.s, -1)
        self.assertEqual(result.iloc[0], 2.0)
        self.assertTrue(np.isnan(result.iloc[-1]))

    # -- DataFrame --

    def test_dataframe_offset(self):
        result = ta.utils.apply_offset(self.df, 1)
        self.assertIsInstance(result, DataFrame)
        self.assertTrue(np.isnan(result["a"].iloc[0]))
        self.assertEqual(result["a"].iloc[1], 1.0)

    # -- list of Series --

    def test_list_offset(self):
        s2 = Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = ta.utils.apply_offset([self.s, s2], 1)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result[0].iloc[0]))
        self.assertTrue(np.isnan(result[1].iloc[0]))
        self.assertEqual(result[0].iloc[1], 1.0)
        self.assertEqual(result[1].iloc[1], 10.0)

    def test_list_unpack(self):
        s2 = Series([10.0, 20.0, 30.0])
        a, b = ta.utils.apply_offset([self.s, s2], 0)
        self.assertIs(a, self.s)
        self.assertIs(b, s2)


class TestApplyFill(TestCase):
    """Unit tests for apply_fill (utils/_core.py)."""

    def setUp(self):
        self.s = Series([np.nan, 2.0, np.nan, 4.0, np.nan], name="test")
        self.df = DataFrame(
            {
                "a": [np.nan, 2.0, np.nan],
                "b": [1.0, np.nan, 3.0],
            }
        )

    # -- no-op case --

    def test_no_kwargs_is_noop(self):
        original_nans = self.s.isna().sum()
        result = ta.utils.apply_fill(self.s)
        self.assertEqual(result.isna().sum(), original_nans)

    # -- fillna --

    def test_fillna_value(self):
        result = ta.utils.apply_fill(self.s, fillna=0)
        self.assertEqual(result.isna().sum(), 0)
        self.assertEqual(result.iloc[0], 0)
        self.assertEqual(result.iloc[2], 0)

    def test_fillna_preserves_existing(self):
        result = ta.utils.apply_fill(self.s, fillna=-1)
        self.assertEqual(result.iloc[1], 2.0)
        self.assertEqual(result.iloc[3], 4.0)

    # -- fill_method ffill --

    def test_ffill(self):
        s = Series([1.0, np.nan, np.nan, 4.0, np.nan])
        result = ta.utils.apply_fill(s, fill_method="ffill")
        self.assertEqual(result.iloc[1], 1.0)
        self.assertEqual(result.iloc[2], 1.0)
        # Leading NaN stays NaN (nothing to forward-fill from)
        self.assertTrue(np.isnan(ta.utils.apply_fill(Series([np.nan, 2.0]), fill_method="ffill").iloc[0]))

    # -- fill_method bfill --

    def test_bfill(self):
        s = Series([np.nan, np.nan, 3.0, np.nan, 5.0])
        result = ta.utils.apply_fill(s, fill_method="bfill")
        self.assertEqual(result.iloc[0], 3.0)
        self.assertEqual(result.iloc[1], 3.0)
        # Trailing NaN stays NaN (nothing to back-fill from)
        self.assertTrue(np.isnan(ta.utils.apply_fill(Series([1.0, np.nan]), fill_method="bfill").iloc[-1]))

    # -- unknown fill_method is ignored --

    def test_unknown_fill_method_raises(self):
        # an unknown method used to be ignored, leaving the NaNs in place
        with self.assertRaisesRegex(ValueError, r"fill_method must be one of \['bfill', 'ffill'\], got 'bogus'"):
            ta.utils.apply_fill(self.s, fill_method="bogus")

    # -- fillna + fill_method combined --

    def test_fillna_then_fill_method(self):
        # fillna runs first, fill_method second — both should work
        s = Series([np.nan, 2.0, np.nan])
        result = ta.utils.apply_fill(s, fillna=0, fill_method="ffill")
        self.assertEqual(result.isna().sum(), 0)
        self.assertEqual(result.iloc[0], 0)

    # -- DataFrame --

    def test_dataframe_fillna(self):
        result = ta.utils.apply_fill(self.df, fillna=0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_dataframe_ffill(self):
        df = DataFrame({"x": [1.0, np.nan, 3.0]})
        result = ta.utils.apply_fill(df, fill_method="ffill")
        self.assertEqual(result["x"].iloc[1], 1.0)

    # -- list of Series --

    def test_list_fillna(self):
        s2 = Series([np.nan, 10.0])
        result = ta.utils.apply_fill([self.s, s2], fillna=0)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].isna().sum(), 0)
        self.assertEqual(result[1].isna().sum(), 0)

    def test_list_ffill(self):
        s1 = Series([1.0, np.nan])
        s2 = Series([np.nan, 2.0])
        a, b = ta.utils.apply_fill([s1, s2], fill_method="ffill")
        self.assertEqual(a.iloc[1], 1.0)
        self.assertTrue(np.isnan(b.iloc[0]))  # nothing to ffill from


class TestVerifySeries(TestCase):
    """verify_series() on a 4-row Series: None below min_length, the Series otherwise, and its argument checks."""

    @classmethod
    def setUpClass(cls):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        cls.c = Series([10.0, 11.0, 10.5, 11.5], index=idx)
        cls.o = Series([9.5, 10.5, 10.0, 11.0], index=idx)
        cls.h = Series([10.5, 11.5, 11.0, 12.0], index=idx)
        cls.l = Series([9.0, 10.0, 9.5, 10.5], index=idx)
        cls.v = Series([1e6, 1.1e6, 9e5, 1.2e6], index=idx)

    @classmethod
    def tearDownClass(cls):
        del cls.c, cls.o, cls.h, cls.l, cls.v

    # ---- verify_series (core utility) ----

    def test_verify_series_none_when_too_short(self):
        result = ta.utils.verify_series(self.c, 10)
        self.assertIsNone(result)

    def test_verify_series_ok_when_long_enough(self):
        result = ta.utils.verify_series(self.c, 4)
        self.assertIsInstance(result, Series)

    def test_verify_series_applies_a_float_min_length(self):
        # it used to be ignored: verify_series(4 rows, 4.5) returned the series
        self.assertIsNone(ta.utils.verify_series(self.c, 4.5))
        self.assertIsInstance(ta.utils.verify_series(self.c, 3.5), Series)
        # numpy numbers were ignored the same way (np.int64 is not an int)
        self.assertIsNone(ta.utils.verify_series(self.c, np.int64(5)))

    def test_verify_series_rejects_an_invalid_min_length(self):
        # NaN, a negative number and a bool passed silently (size < nan is False),
        # inf blanked the result and a string raised a bare TypeError from "<"
        for bad in (np.nan, -3, np.inf, True, "5"):
            with self.subTest(min_length=bad), self.assertRaisesRegex(ValueError, r"^verify_series\(\) min_length must be a number >= 0"):
                ta.utils.verify_series(self.c, bad)

    def test_verify_series_non_series_raises_type_error(self):
        """Issue #145 case (A): non-Series input is a caller error.

        0.8.32 returned None with a FutureWarning; it now raises TypeError
        naming the indicator and the type it received.
        """
        for bad in (self.c.to_numpy(), list(self.c), self.c.to_frame(), 5, "close"):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(TypeError) as ctx:
                    ta.sma(bad, 3)
                self.assertEqual(
                    str(ctx.exception),
                    f"sma() expected a pandas Series but got {type(bad).__name__}. Pass a Series, e.g. df['close'] rather than df['close'].values.",
                )

    def test_verify_series_none_and_short_series_do_not_raise(self):
        """Case (B) and omitted optional arguments return None without warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            self.assertIsNone(ta.utils.verify_series(None))
            self.assertIsNone(ta.utils.verify_series(self.c, 10))
            self.assertIsNotNone(ta.ad(self.h, self.l, self.c, Series([1.0] * len(self.c), index=self.c.index), open_=None))


class TestRequiredUtilityInputs(TestCase):
    """A required input passed as None raises ValueError naming the function and the argument.

    These helpers called a pandas method on it and failed with
    ``'NoneType' object has no attribute 'diff'`` (or ``index``, ``copy``,
    ``size``), naming neither. The metrics that answer None with NaN, and the
    signal helpers that answer it with None, are a separate contract and are
    not listed here.
    """

    def test_signals_names_a_missing_indicator(self):
        # signals takes keyword options the table below leaves out.
        s = Series(np.linspace(10.0, 20.0, 30), index=pd.date_range("2020-01-01", periods=30, freq="D"))
        u = ta.utils
        signals_args = {
            "xa": 15,
            "xb": None,
            "cross_values": False,
            "xserie": None,
            "xserie_a": None,
            "xserie_b": None,
            "cross_series": True,
            "offset": None,
        }
        u.signals(s, **signals_args)
        with self.assertRaisesRegex(ValueError, r"^signals\(\) requires 'indicator', got None$"):
            u.signals(None, **signals_args)

    def test_np_rolling_moments_rejects_a_length_below_one(self):
        # length=0 used to fail inside numpy: could not broadcast input array from shape (11,) into shape (1,)
        for length in (0, -1, 2.5):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, r"^np_rolling_moments\(\) length must be"):
                ta.utils.np_rolling_moments(np.arange(10.0), length, 2)

    def test_np_rolling_moments_rejects_a_min_periods_outside_1_to_length(self):
        # min_periods=0 or -3 started the partial-window loop at -1 and wrote the last bar
        for min_periods in (0, -3, 6, 2.5):
            with self.subTest(min_periods=min_periods), self.assertRaisesRegex(ValueError, r"^np_rolling_moments\(\) min_periods must be"):
                ta.utils.np_rolling_moments(np.arange(10.0), 5, 2, min_periods=min_periods)

    def test_every_utility_names_a_required_input_passed_as_none(self):
        """Every public utility is either in the table below or exempt, so a new one cannot be missed."""
        s = Series(np.linspace(10.0, 20.0, 30), index=pd.date_range("2020-01-01", periods=30, freq="D"))
        df = DataFrame({"close": s})
        # The metrics need a series that falls as well as rises (a drawdown, a downside).
        wave = s * (1 + 0.05 * np.sin(np.arange(30)))
        # A valid call for each utility with required inputs; each input in turn is replaced by None.
        valid_calls = {
            "cagr": {"close": wave},
            "calmar_ratio": {"close": wave},
            "downside_deviation": {"returns": wave.pct_change()},
            "jensens_alpha": {"returns": wave.pct_change(), "benchmark_returns": s.pct_change()},
            "log_max_drawdown": {"close": wave},
            "max_drawdown": {"close": wave},
            "optimal_leverage": {"close": wave},
            "pure_profit_score": {"close": wave},
            "sharpe_ratio": {"close": wave},
            "sortino_ratio": {"close": wave},
            "candle_color": {"open_": s, "close": s},
            "degenerate_div": {"numerator": s, "denominator": s},
            "degenerate_zero": {"x": s},
            # annotated DataFrame, but only Series work: DataFrame.corr() takes no other frame
            "df_error_analysis": {"dfA": s, "dfB": s},
            "df_year_to_date": {"df": df},
            "final_time": {"stime": 0.0},
            "is_datetime_ordered": {"df": df},
            "linear_regression": {"x": s, "y": s},
            "np_rolling_moments": {"values": s.to_numpy(), "length": 5},
            "recent_maximum_index": {"x": s.to_numpy()},
            "recent_minimum_index": {"x": s.to_numpy()},
            "signed_series": {"series": s},
            "to_utc": {"df": df},
            "total_time": {"df": df},
            "unsigned_differences": {"series": s},
            "weights": {"w": np.ones(5)},
        }
        exempt = {
            # No required input at all.
            "combination", "fibonacci", "pascals_triangle", "symmetric_triangle",
            # Covered by test_signals_names_a_missing_indicator: its options need values of their own.
            "signals",
            # None selects the default.
            "get_drift", "get_offset",
            # A predicate: None is not a percent.
            "is_percent",
            # Raises TypeError naming the function for any non-str, None included.
            "tal_ma",
            # Series.apply runs it once per element (adx, dm, psar, cross); a guard there costs every bar.
            "zero",
            # None in, None out: an indicator's missing optional input flows through.
            "apply_fill", "apply_offset", "verify_series", "above", "above_value", "below", "below_value",
            "cross", "cross_value", "crossover", "lag",
        }  # fmt: skip
        public = {name for name in ta.utils.__all__ if inspect.isfunction(getattr(ta.utils, name))}
        self.assertEqual(public - exempt, set(valid_calls), "add the new utility to valid_calls or to exempt, with the reason")
        self.assertFalse(exempt - public, "an exempt name is not a public utility any more")

        for name, inputs in valid_calls.items():
            func = getattr(ta.utils, name)
            params = inspect.signature(func).parameters.values()
            required = {p.name for p in params if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
            self.assertEqual(set(inputs), required, f"{name}: list every required input in valid_calls")
            func(**inputs)  # the valid call works, so None is the only thing wrong below
            for arg in inputs:
                with self.subTest(func=name, argument=arg), self.assertRaisesRegex(ValueError, rf"^{name}\(\) requires '{arg}', got None$"):
                    func(**{**inputs, arg: None})


class TestNonZeroRangeRemoved(TestCase):
    """non_zero_range() was removed in 0.9.0; both access paths name the replacement."""

    def test_attribute_access_raises_with_the_recipe(self):
        for owner in (ta, ta.utils):
            with self.subTest(owner=owner.__name__), self.assertRaisesRegex(AttributeError, r"non_zero_range\(\) was removed in 0\.9\.0"):
                owner.non_zero_range  # noqa: B018

    def test_import_raises(self):
        with self.assertRaises(ImportError):
            from pandas_ta_classic.utils import non_zero_range  # noqa: F401


# ---------------------------------------------------------------------------
# is_datetime_ordered returns bool (no return-in-finally SyntaxWarning)
# ---------------------------------------------------------------------------


class TestIsDatetimeOrdered(TestCase):
    """is_datetime_ordered must return a plain bool in all code paths."""

    def _fn(self):
        from pandas_ta_classic.utils import is_datetime_ordered

        return is_datetime_ordered

    def test_ordered_datetime_index_returns_true(self):
        fn = self._fn()
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series(range(5), index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_reverse_datetime_index_returns_false(self):
        fn = self._fn()
        idx = pd.date_range("2020-01-01", periods=5, freq="D")[::-1]
        s = pd.Series(range(5), index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_non_datetime_index_returns_false(self):
        fn = self._fn()
        s = pd.Series(range(5))  # integer index
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_single_element_returns_false(self):
        fn = self._fn()
        idx = pd.DatetimeIndex(["2020-01-01"])
        s = pd.Series([1.0], index=idx)
        result = fn(s)
        self.assertIsInstance(result, bool)
        self.assertFalse(result)


# Shorthand strategies
_positive_int = st.integers(min_value=1, max_value=100)


# ======================================================================
# Core utility properties
# ======================================================================


class TestVerifySeriesProperties(TestCase):
    """Property tests for ``verify_series``."""

    @given(price_series(min_size=2, max_size=200, allow_nan=False))
    def test_verify_series_returns_series(self, s):
        result = verify_series(s, min_length=1)
        assert isinstance(result, pd.Series)

    @given(price_series(min_size=10, max_size=200, allow_nan=False))
    def test_verify_series_preserves_name(self, s):
        s.name = "test_col"
        result = verify_series(s, min_length=5)
        assert result.name == "test_col"

    @given(
        price_series(min_size=1, max_size=50, allow_nan=False),
        _positive_int,
    )
    def test_verify_series_short_returns_none(self, s, min_len):
        assume(len(s) < min_len)
        result = verify_series(s, min_length=min_len)
        assert result is None

    @given(
        price_series(min_size=5, max_size=50, allow_nan=False),
        _positive_int,
    )
    def test_verify_series_long_enough_not_none(self, s, min_len):
        assume(len(s) >= min_len)
        result = verify_series(s, min_length=min_len)
        assert result is not None
        assert len(result) == len(s)

    @given(price_series(min_size=1, max_size=100, allow_nan=False))
    def test_verify_series_none_input_returns_none(self, s):
        """Passing None directly must return None."""
        assert verify_series(None, min_length=1) is None

    @given(price_series(min_size=1, max_size=100, allow_nan=True, nan_prob=0.2), st.sampled_from(["Float64", "Float32", "Int64"]))
    def test_verify_series_turns_nullable_numeric_into_float64(self, s, dtype):
        """A nullable Series comes back as float64, ``pd.NA`` as NaN, index and name kept, input untouched."""
        if dtype == "Int64":
            s = s.round()
        s.name = "close"
        nullable = s.astype(dtype)
        before = nullable.copy()
        result = verify_series(nullable, min_length=1)
        assert result.dtype == np.float64
        pd.testing.assert_series_equal(result, nullable.astype(float))
        assert result.index is nullable.index
        pd.testing.assert_series_equal(nullable, before)


_offset_int = st.integers(min_value=0, max_value=20)


class TestApplyOffsetProperties(TestCase):
    """Property tests for ``apply_offset``."""

    @given(price_series(min_size=10, max_size=100, allow_nan=False), _offset_int)
    def test_apply_offset_preserves_length(self, s, offset):
        result = apply_offset(s, offset)
        assert len(result) == len(s)

    @given(price_series(min_size=10, max_size=100, allow_nan=False))
    def test_apply_offset_zero_is_identity(self, s):
        result = apply_offset(s, 0)
        pd.testing.assert_series_equal(result, s)

    @given(price_series(min_size=10, max_size=100, allow_nan=False), _offset_int)
    def test_apply_offset_shift_direction(self, s, offset):
        assume(offset >= 2 and len(s) > offset + 1)
        result = apply_offset(s, offset)
        # First `offset` rows should be NaN after shift
        assert result.iloc[:offset].isna().all()


class TestApplyFillProperties(TestCase):
    """Property tests for ``apply_fill``."""

    @given(price_series(min_size=10, max_size=100, allow_nan=True, nan_prob=0.3))
    def test_apply_fill_fillna_replaces_nan(self, s):
        filled = apply_fill(s.copy(), fillna=0.0)
        assert not filled.isna().any()

    @given(price_series(min_size=10, max_size=100, allow_nan=True, nan_prob=0.3))
    def test_apply_fill_ffill_no_gaps(self, s):
        assume(s.iloc[0] is not None and not np.isnan(s.iloc[0]))
        filled = apply_fill(s.copy(), fill_method="ffill")
        # After ffill, only a leading-NaN gap can remain
        first_valid = filled.first_valid_index()
        if first_valid is not None and first_valid > 0:
            assert filled.iloc[:first_valid].isna().all()
        assert not filled.iloc[first_valid:].isna().any() if first_valid is not None else True


class TestOffsetDriftProperties(TestCase):
    """Properties of get_offset and get_drift."""

    @given(st.floats(allow_nan=True, allow_infinity=True).filter(lambda v: not (math.isfinite(v) and v.is_integer())))
    def test_get_offset_non_integral_raises(self, x):
        """Fractional, NaN and infinite offsets raise instead of becoming 0."""
        with pytest.raises(ValueError, match="offset must be an integer"):
            get_offset(x)

    @given(st.integers(min_value=-100, max_value=100))
    def test_get_offset_int_identity(self, x):
        result = get_offset(x)
        assert isinstance(result, int)
        assert result == x

    @given(st.integers(min_value=1, max_value=100))
    def test_get_drift_positive_identity(self, x):
        assert get_drift(x) == x
        assert get_drift(float(x)) == x

    @given(st.integers(min_value=-100, max_value=0))
    def test_get_drift_non_positive_raises(self, x):
        """Zero and negative drifts raise instead of becoming 1."""
        with pytest.raises(ValueError, match="drift must be an integer > 0"):
            get_drift(x)
