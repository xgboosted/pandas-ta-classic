"""Tests of pandas_ta_classic/utils/_math.py: combinatorics, weights, linear regression, rolling moments."""

import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.extra.numpy import arrays
from pandas import Series

import pandas_ta_classic as pandas_ta
from tests.strategies import constant_price_series, price_series


class TestMathUtilities:

    def setup_method(self):
        self.utils = pandas_ta.utils

    def teardown_method(self):
        del self.utils

    def test_combination(self):
        assert self.utils.combination() is not None

        assert self.utils.combination() == 1
        with pytest.raises(ValueError, match=r"combination\(\) r must be an integer >= 0, got -1"):
            self.utils.combination(r=-1)
        # choosing 0 of 0 kinds with repetition: one way (math.comb(-1, 0) raised)
        assert self.utils.combination(n=0, r=0, repetition=True) == 1

        assert self.utils.combination(n=10, r=4, repetition=False) == 210
        assert self.utils.combination(n=10, r=4, repetition=True) == 715
        # the multichoose alias of repetition was removed in 0.9.0
        with pytest.raises(TypeError, match="multichoose"):
            self.utils.combination(n=10, r=4, multichoose=True)
        with pytest.raises(TypeError, match=r"unexpected keyword argument 'repetiton'"):
            self.utils.combination(n=10, r=4, repetiton=True)

    def test_helpers_reject_misspelled_keywords(self, subtests):
        """A misspelled option used to be ignored and the default result returned.

        fibonacci(n=3, weigthed=True) came back unweighted, with no error.
        """
        calls = {
            "fibonacci": lambda: self.utils.fibonacci(n=3, weigthed=True),
            "pascals_triangle": lambda: self.utils.pascals_triangle(n=3, weigthed=True),
            "symmetric_triangle": lambda: self.utils.symmetric_triangle(n=3, weigthed=True),
            "unsigned_differences": lambda: self.utils.unsigned_differences(Series([1.0, 2.0, 1.0]), asnit=True),
            "df_error_analysis": lambda: self.utils.df_error_analysis(Series([1.0, 2.0]), Series([1.0, 2.0]), corr_metod="spearman"),
        }
        for name, call in calls.items():
            with subtests.test(name=name), pytest.raises(TypeError, match="unexpected keyword argument"):
                call()

    def test_fibonacci(self):
        assert type(self.utils.fibonacci(zero=True, weighted=False)) is np.ndarray

        np.testing.assert_array_equal(self.utils.fibonacci(zero=True), np.array([0, 1, 1]))
        np.testing.assert_array_equal(self.utils.fibonacci(zero=False), np.array([1, 1]))

        np.testing.assert_array_equal(self.utils.fibonacci(n=0, zero=True, weighted=False), np.array([0]))
        np.testing.assert_array_equal(self.utils.fibonacci(n=0, zero=False, weighted=False), np.array([1]))

        np.testing.assert_array_equal(
            self.utils.fibonacci(n=5, zero=True, weighted=False),
            np.array([0, 1, 1, 2, 3, 5]),
        )
        np.testing.assert_array_equal(
            self.utils.fibonacci(n=5, zero=False, weighted=False),
            np.array([1, 1, 2, 3, 5]),
        )

    def test_fibonacci_weighted(self):
        assert type(self.utils.fibonacci(zero=True, weighted=True)) is np.ndarray
        np.testing.assert_array_equal(self.utils.fibonacci(n=0, zero=True, weighted=True), np.array([0]))
        np.testing.assert_array_equal(self.utils.fibonacci(n=0, zero=False, weighted=True), np.array([1]))

        np.testing.assert_allclose(
            self.utils.fibonacci(n=5, zero=True, weighted=True),
            np.array([0, 1 / 12, 1 / 12, 1 / 6, 1 / 4, 5 / 12]),
        )
        np.testing.assert_allclose(
            self.utils.fibonacci(n=5, zero=False, weighted=True),
            np.array([1 / 12, 1 / 12, 1 / 6, 1 / 4, 5 / 12]),
        )

    def test_fibonacci_weights_do_not_overflow(self, subtests):
        """The int64 seed array used to wrap at n=91.

        fibonacci(91, weighted=True) summed to -6.2e18, so the weights came out
        with the wrong sign and magnitude and fwma(length=91) returned nonsense
        — between two lengths that both looked fine.
        """
        for n in (10, 88, 89, 90, 91, 92, 93, 200, 1500):
            with subtests.test(n=n):
                weights = self.utils.fibonacci(n=n, weighted=True)
                assert weights.size == n
                assert round(abs(float(weights.sum()) - 1.0), 12) == 0
                assert (weights > 0).all()
                # Fibonacci weights rise monotonically towards the last bar.
                assert (np.diff(weights) >= 0).all()

        # Unweighted stays an exact integer sequence past 2**63 instead of
        # wrapping; object dtype is what carries the big terms.
        big = self.utils.fibonacci(n=95)
        assert int(big[-1]) == int(big[-2]) + int(big[-3])
        assert int(big[-1]) > 2**63

    def test_linear_regression(self):
        x = Series([1, 2, 3, 4, 5])
        y = Series([1.8, 2.1, 2.7, 3.2, 4])

        result = self.utils.linear_regression(x, y)
        assert isinstance(result, dict)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], float)
        assert isinstance(result["r"], float)
        assert isinstance(result["t"], float)
        assert isinstance(result["line"], Series)

        with pytest.raises(ValueError, match=r"must have equal length"):
            self.utils.linear_regression(x, y.iloc[:-1])

    def test_linear_regression_slope_survives_a_far_from_zero_x(self, subtests):
        # m * sum(x * y) - sum(x) * sum(y) cancelled: at x ~ 1e6 with a 1e-3
        # spread the slope was 99% off, at 1e8 its sign flipped.
        rng = np.random.default_rng(0)
        for level, spread in [(1e4, 1e-3), (1e6, 1e-3), (1e8, 1e-2)]:
            with subtests.test(level=level):
                x = Series(level + rng.normal(0, spread, 300))
                y = 2.0 * (x - level) / spread + rng.normal(0, 0.1, 300)
                expected = np.polyfit(x - x.mean(), y, 1)[0]
                assert round(abs((self.utils.linear_regression(x, y)["b"] / expected) - 1.0), 10) == 0

    def test_linear_regression_near_flat_x_is_undefined(self):
        # 0.1 + 0.2 and 0.3 differ by one ULP: x.std() is ~4e-17, not 0, so the
        # old `!= 0` guard let the slope divide by it and read inf.
        x = Series([0.1 + 0.2] * 150 + [0.3] * 150)
        y = Series(np.random.default_rng(1).normal(0, 1, 300))
        result = self.utils.linear_regression(x, y)
        for key in ("a", "b", "r", "t"):
            assert np.isnan(result[key]), f"{key} = {result[key]}"

    def test_linear_regression_flat_x_is_undefined_at_any_level(self, subtests):
        # One ULP apart at 1e6 the deviation is ~8e-11, past an absolute 1e-12
        # tolerance: the slope read -1.8e8 instead of NaN.
        y = Series(np.random.default_rng(1).normal(0, 1, 300))
        for level in (0.3, 1e3, 1e6, 1e9):
            with subtests.test(level=level):
                x = Series([level] * 150 + [np.nextafter(level, np.inf)] * 150)
                assert np.isnan(self.utils.linear_regression(x, y)["b"])

    def test_linear_regression_small_scale_x_is_not_flat(self):
        # A real spread below 1e-12 is not a residue when x itself is that small.
        x = Series(np.linspace(0.0, 1e-13, 300))
        assert round(abs((self.utils.linear_regression(x, 2.0 * x / 1e-13)["b"] * 1e-13) - 2.0), 8) == 0

    def test_linear_regression_flat_y_has_a_slope_but_no_correlation(self):
        x = Series(np.arange(300.0))
        y = Series(5.0, index=x.index)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # np.corrcoef warned "invalid value encountered in divide"
            result = self.utils.linear_regression(x, y)
            # A perfect fit has 1 - r * r = 0: t is inf, the answer, not a warning.
            perfect = self.utils.linear_regression(x, 2.0 * x + 1.0)
        assert round(abs(result["b"] - 0.0), 12) == 0
        assert round(abs(result["a"] - 5.0), 12) == 0
        assert (np.isnan(result["r"]) and np.isnan(result["t"]))
        assert round(abs(perfect["b"] - 2.0), 12) == 0
        assert perfect["t"] > 1e6

    def test_linear_regression_rejects_what_it_cannot_fit(self, subtests):
        # Two points gave t = 0.0 (no degrees of freedom), one or none gave NaN
        # with numpy warnings, and a NaN gave a wrong slope: the sums skipped it
        # while the point count did not (x = [0, 1, NaN, 3, 4] fitted b = 0.758).
        for size in (0, 1, 2):
            with subtests.test(size=size), pytest.raises(ValueError, match=rf"needs at least 3 points, got {size}"):
                self.utils.linear_regression(Series(range(size), dtype=float), Series(range(size), dtype=float))
        line = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        gappy = Series([0.0, 1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match=r"x has 1 missing value"):
            self.utils.linear_regression(gappy, line)
        with pytest.raises(ValueError, match=r"y has 1 missing value"):
            self.utils.linear_regression(line, gappy)
        # x * y aligned on two disjoint indexes and fitted y = 2x + 1 as b = -10.5.
        with pytest.raises(ValueError, match=r"must share the same index"):
            self.utils.linear_regression(line, (2 * line + 1).set_axis(list("abcde")))

    def test_pascals_triangle(self):
        with pytest.raises(ValueError, match="inverse=True needs weighted=True"):  # used to return None
            self.utils.pascals_triangle(inverse=True)

        array_1 = np.array([1])
        np.testing.assert_array_equal(self.utils.pascals_triangle(), array_1)
        np.testing.assert_array_equal(self.utils.pascals_triangle(weighted=True), array_1)
        np.testing.assert_array_equal(self.utils.pascals_triangle(weighted=True, inverse=True), np.array([0]))

        array_5 = self.utils.pascals_triangle(n=5)  # or np.array([1, 5, 10, 10, 5, 1])
        array_5w = array_5 / np.sum(array_5)
        array_5iw = 1 - array_5w
        with pytest.raises(ValueError, match=r"pascals_triangle\(\) n must be an integer >= 0, got -5"):
            self.utils.pascals_triangle(n=-5)

        np.testing.assert_array_equal(self.utils.pascals_triangle(n=5), array_5)
        np.testing.assert_array_equal(self.utils.pascals_triangle(n=5, weighted=True), array_5w)
        np.testing.assert_array_equal(self.utils.pascals_triangle(n=5, weighted=True, inverse=True), array_5iw)

    def test_symmetric_triangle(self):
        np.testing.assert_array_equal(self.utils.symmetric_triangle(), np.array([1, 1]))
        np.testing.assert_array_equal(self.utils.symmetric_triangle(weighted=True), np.array([0.5, 0.5]))

        array_4 = self.utils.symmetric_triangle(n=4)  # or np.array([1, 2, 2, 1])
        array_4w = array_4 / np.sum(array_4)
        np.testing.assert_array_equal(self.utils.symmetric_triangle(n=4), array_4)
        np.testing.assert_array_equal(self.utils.symmetric_triangle(n=4, weighted=True), array_4w)

        array_5 = self.utils.symmetric_triangle(n=5)  # or np.array([1, 2, 3, 2, 1])
        array_5w = array_5 / np.sum(array_5)
        np.testing.assert_array_equal(self.utils.symmetric_triangle(n=5), array_5)
        np.testing.assert_array_equal(self.utils.symmetric_triangle(n=5, weighted=True), array_5w)

    def test_zero(self):
        assert self.utils.zero(-0.0000000000000001) == 0
        assert self.utils.zero(0) == 0
        assert self.utils.zero(0.0) == 0
        assert self.utils.zero(0.0000000000000001) == 0

        assert self.utils.zero(-0.000000000000001) != 0
        assert self.utils.zero(0.000000000000001) != 0
        assert self.utils.zero(1) != 0


class TestNpRollingMoments:
    """Unit tests for the np_rolling_moments helper added in the numpy-
    determinism PR (utils/_math.py)."""

    def setup_method(self):
        from pandas_ta_classic.utils import np_rolling_moments

        self.fn = np_rolling_moments
        # Simple increasing array: [0, 1, 2, 3, 4]
        self.arr = np.arange(5, dtype=float)

    # ------------------------------------------------------------------
    # Return-type / shape
    # ------------------------------------------------------------------

    def test_returns_tuple_of_arrays(self):
        result = self.fn(self.arr, 3, 2)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_output_length_matches_input(self):
        (m2,) = self.fn(self.arr, 3, 2)
        assert len(m2) == len(self.arr)

    def test_multiple_orders(self):
        m2, m3, m4 = self.fn(self.arr, 3, 2, 3, 4)
        for arr in (m2, m3, m4):
            assert len(arr) == len(self.arr)

    def test_output_dtype_is_float64(self):
        (m2,) = self.fn(self.arr, 3, 2)
        assert m2.dtype == np.float64

    # ------------------------------------------------------------------
    # Default min_periods == length: first (length-1) slots are NaN
    # ------------------------------------------------------------------

    def test_default_min_periods_leading_nans(self):
        length = 3
        (m2,) = self.fn(self.arr, length, 2)
        assert np.all(np.isnan(m2[: length - 1]))

    def test_default_min_periods_no_trailing_nans(self):
        length = 3
        (m2,) = self.fn(self.arr, length, 2)
        assert not np.any(np.isnan(m2[length - 1 :]))

    # ------------------------------------------------------------------
    # Correctness of computed sums (order-2 == sum of squared deviations)
    # ------------------------------------------------------------------

    def test_order2_known_window(self):
        # window [1, 2, 3]: mean=2, deviations=[-1,0,1], sum = 2
        (m2,) = self.fn(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3, 2)
        np.testing.assert_allclose(m2[2], 2.0)  # first full window [1,2,3]
        np.testing.assert_allclose(m2[3], 2.0)  # [2,3,4]
        np.testing.assert_allclose(m2[4], 2.0)  # [3,4,5]

    def test_order2_constant_series_is_zero(self):
        arr = np.ones(6)
        (m2,) = self.fn(arr, 3, 2)
        np.testing.assert_allclose(m2[2:], 0.0, atol=1e-12)

    def test_order3_sign(self):
        # Positively skewed window: [1, 2, 10] - third central moment > 0
        arr = np.array([1.0, 2.0, 10.0])
        (m3,) = self.fn(arr, 3, 3)
        assert m3[2] > 0

    # ------------------------------------------------------------------
    # min_periods < length  (partial-window behaviour)
    # ------------------------------------------------------------------

    def test_min_periods_position_below_min_is_nan(self):
        # min_periods=2 -> position 0 (only 1 value) must be NaN
        # length is positional so the order value (2) lands in *orders
        (m2,) = self.fn(self.arr, 4, 2, min_periods=2)
        assert np.isnan(m2[0])

    def test_min_periods_first_valid_position(self):
        # min_periods=2 -> position 1 uses window [0, 1]: deviations [-0.5, 0.5]
        # sum of squares = 0.5
        (m2,) = self.fn(self.arr, 4, 2, min_periods=2)
        np.testing.assert_allclose(m2[1], 0.5)

    def test_min_periods_full_window_unchanged(self):
        # Positions >= length-1 must be identical regardless of min_periods
        m2_default = self.fn(self.arr, 3, 2)[0]
        m2_partial = self.fn(self.arr, 3, 2, min_periods=2)[0]
        np.testing.assert_array_equal(m2_default[2:], m2_partial[2:])

    def test_min_periods_equal_to_length_same_as_default(self):
        m2_default = self.fn(self.arr, 3, 2)[0]
        m2_explicit = self.fn(self.arr, 3, 2, min_periods=3)[0]
        np.testing.assert_array_equal(m2_default, m2_explicit)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_series_shorter_than_length_all_nan(self):
        arr = np.array([1.0, 2.0])
        (m2,) = self.fn(arr, 5, 2)
        assert np.all(np.isnan(m2))

    def test_empty_array(self):
        arr = np.array([], dtype=float)
        (m2,) = self.fn(arr, 3, 2)
        assert len(m2) == 0

    def test_single_element_series(self):
        arr = np.array([42.0])
        (m2,) = self.fn(arr, 1, 2)
        # Window of size 1: deviation is 0, sum = 0
        np.testing.assert_allclose(m2[0], 0.0, atol=1e-12)


def _exact_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope in exact rational arithmetic: a reference without rounding."""
    fx, fy = [Fraction(float(v)) for v in xs], [Fraction(float(v)) for v in ys]
    x_mean, y_mean = sum(fx) / len(fx), sum(fy) / len(fy)
    sxy = sum((a - x_mean) * (b - y_mean) for a, b in zip(fx, fy))
    sxx = sum((a - x_mean) ** 2 for a in fx)
    return float(sxy / sxx)


class TestLinearRegressionProperties:
    """Property tests for ``linear_regression``.

    The slope used the one-pass ``m * sum(x * y) - sum(x) * sum(y)``, which
    cancels when x sits far from 0 with a small spread (99% off at 1e6, the
    wrong sign at 1e8). Constancy was tested through ``std() != 0``, which
    rounding defeats: a constant x, or one mixing two adjacent floats, fitted a
    made-up slope, and a constant y gave r = NaN or +-1e-16 and made
    ``np.corrcoef`` warn.
    """

    @given(
        st.integers(min_value=3, max_value=40),
        st.floats(min_value=-12.0, max_value=9.0),
        st.sampled_from([1.0, -1.0]),
        st.integers(min_value=1, max_value=9),
        st.data(),
    )
    def test_slope_matches_the_exact_fit_at_any_level(self, n, exponent, sign, digits, data):
        # x = level + a spread `digits` decimal digits below the level.
        level = sign * 10.0**exponent
        unit = abs(level) * 10.0**-digits
        offsets = data.draw(arrays(np.float64, n, elements=st.floats(min_value=-1.0, max_value=1.0, width=64)))
        xs = level + offsets * unit
        assume(np.std(xs) > 1e-10 * abs(level))
        ys = data.draw(arrays(np.float64, n, elements=st.floats(min_value=-1000.0, max_value=1000.0, width=64)))
        expected = _exact_slope(xs, ys)
        # The slope's natural scale is spread(y) / spread(x); measure the error against it.
        scale = abs(expected) + (np.std(ys) + 1.0) / np.std(xs)
        assert abs(pandas_ta.linear_regression(pd.Series(xs), pd.Series(ys))["b"] - expected) <= 1e-9 * scale

    @given(st.integers(min_value=3, max_value=60), st.floats(min_value=-12.0, max_value=9.0), st.sampled_from([1.0, -1.0]), st.data())
    def test_x_of_two_adjacent_floats_has_no_fit(self, n, exponent, sign, data):
        # One ULP apart, x.std() is a rounding residue (8e-11 at 1e6), not a spread.
        level = sign * 10.0**exponent
        upper = data.draw(arrays(np.bool_, n))
        assume(upper.any() and not upper.all())
        x = pd.Series(np.where(upper, np.nextafter(level, np.inf), level))
        y = pd.Series(np.arange(n, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pandas_ta.linear_regression(x, y)
        assert all(np.isnan(result[key]) for key in ("a", "b", "r", "t"))

    @given(constant_price_series(min_size=3, max_size=100))
    def test_constant_x_has_no_fit(self, x):
        y = pd.Series(np.arange(x.size, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pandas_ta.linear_regression(x, y)
        assert all(np.isnan(result[key]) for key in ("a", "b", "r", "t", "line"))

    @given(constant_price_series(min_size=3, max_size=100))
    def test_constant_y_is_quiet(self, y):
        x = pd.Series(np.arange(y.size, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pandas_ta.linear_regression(x, y)
        value = y.iloc[0]
        assert abs(result["b"]) <= 1e-9 * value
        assert result["a"] == pytest.approx(value, rel=1e-9)
        assert np.isnan(result["r"]) and np.isnan(result["t"])

    # The sums skipped NaN while the point count did not, and x * y aligned on the
    # index while np.corrcoef is positional: both gave a wrong slope without an error.
    @given(price_series(min_size=3, max_size=100, allow_nan=False), st.sampled_from(["x", "y"]), st.data())
    def test_nan_anywhere_raises(self, y, side, data):
        x = pd.Series(np.arange(y.size, dtype=float))
        pos = data.draw(st.integers(min_value=0, max_value=y.size - 1))
        args = {"x": x, "y": y.copy()}
        args[side].iloc[pos] = np.nan
        with pytest.raises(ValueError, match=f"{side} has 1 missing value"):
            pandas_ta.linear_regression(args["x"], args["y"])

    @given(price_series(min_size=3, max_size=100, allow_nan=False), st.sampled_from(["x", "y"]), st.sampled_from([np.inf, -np.inf]), st.data())
    def test_infinity_anywhere_raises(self, y, side, value, data):
        # An infinite value made every result NaN, with numpy warnings and no error.
        x = pd.Series(np.arange(y.size, dtype=float))
        pos = data.draw(st.integers(min_value=0, max_value=y.size - 1))
        args = {"x": x, "y": y.copy()}
        args[side].iloc[pos] = value
        with pytest.raises(ValueError, match=f"{side} has 1 infinite value"):
            pandas_ta.linear_regression(args["x"], args["y"])

    @given(price_series(min_size=3, max_size=100, allow_nan=False), st.data())
    def test_reordered_index_raises(self, y, data):
        order = data.draw(st.permutations(range(y.size)))
        assume(list(order) != list(range(y.size)))
        x = pd.Series(np.arange(y.size, dtype=float))
        with pytest.raises(ValueError, match="must share the same index"):
            pandas_ta.linear_regression(x, y.set_axis(y.index[list(order)]))

    @given(price_series(min_size=3, max_size=100, allow_nan=False), st.integers(min_value=1, max_value=1000))
    def test_shifted_index_raises(self, y, shift):
        x = pd.Series(np.arange(y.size, dtype=float))
        with pytest.raises(ValueError, match="must share the same index"):
            pandas_ta.linear_regression(x, y.set_axis(y.index + shift))

    @given(price_series(min_size=3, max_size=100, allow_nan=False), st.integers(min_value=-1000, max_value=1000))
    def test_result_ignores_a_shared_relabelling(self, y, shift):
        x = pd.Series(np.arange(y.size, dtype=float))
        before = pandas_ta.linear_regression(x, y)
        after = pandas_ta.linear_regression(x.set_axis(x.index + shift), y.set_axis(y.index + shift))
        for key in ("a", "b", "r", "t"):
            np.testing.assert_equal(after[key], before[key])
        np.testing.assert_array_equal(after["line"].to_numpy(), before["line"].to_numpy())
