"""Tests of pandas_ta_classic/utils/_signals.py: above/below, cross, signal gaps."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, strategies as st
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.contracts import spy as _spy

data = {
    "zero": [0, 0],
    "a": [0, 1],
    "b": [1, 0],
    "c": [1, 1],
    "crossed": [0, 1],
}


class TestSignalUtilities:

    def setup_method(self):
        self.crosseddf = DataFrame(data)
        self.utils = pandas_ta.utils

    def teardown_method(self):
        del self.crosseddf
        del self.utils

    def test__above_below(self):
        result = self.utils.above(self.crosseddf["a"], self.crosseddf["zero"])
        assert isinstance(result, Series)
        assert result.name == "a_A_zero"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

        result = self.utils.below(self.crosseddf["a"], self.crosseddf["zero"])
        assert isinstance(result, Series)
        assert result.name == "a_B_zero"
        np.testing.assert_array_equal(result, self.crosseddf["b"])

        result = self.utils.above(self.crosseddf["c"], self.crosseddf["zero"])
        assert isinstance(result, Series)
        assert result.name == "c_A_zero"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

        result = self.utils.below(self.crosseddf["c"], self.crosseddf["zero"])
        assert isinstance(result, Series)
        assert result.name == "c_B_zero"
        np.testing.assert_array_equal(result, self.crosseddf["zero"])

    def test_above(self):
        result = self.utils.above(self.crosseddf["a"], self.crosseddf["zero"])
        assert isinstance(result, Series)
        assert result.name == "a_A_zero"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

        result = self.utils.above(self.crosseddf["zero"], self.crosseddf["a"])
        assert isinstance(result, Series)
        assert result.name == "zero_A_a"
        np.testing.assert_array_equal(result, self.crosseddf["b"])

    def test_above_value(self):
        result = self.utils.above_value(self.crosseddf["a"], 0)
        assert isinstance(result, Series)
        assert result.name == "a_A_0"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

        # a non-number value used to log an error and return None
        with pytest.raises(ValueError, match=r"above_value\(\) value must be a number"):
            self.utils.above_value(self.crosseddf["a"], self.crosseddf["zero"])
        assert self.utils.above_value(self.crosseddf["a"], np.int64(0)).name == "a_A_0"

    def test_below(self):
        result = self.utils.below(self.crosseddf["zero"], self.crosseddf["a"])
        assert isinstance(result, Series)
        assert result.name == "zero_B_a"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

        result = self.utils.below(self.crosseddf["zero"], self.crosseddf["a"])
        assert isinstance(result, Series)
        assert result.name == "zero_B_a"
        np.testing.assert_array_equal(result, self.crosseddf["c"])

    def test_below_value(self):
        result = self.utils.below_value(self.crosseddf["a"], 0)
        assert isinstance(result, Series)
        assert result.name == "a_B_0"
        np.testing.assert_array_equal(result, self.crosseddf["b"])

        # a non-number value used to log an error and return None
        with pytest.raises(ValueError, match=r"below_value\(\) value must be a number"):
            self.utils.below_value(self.crosseddf["a"], self.crosseddf["zero"])

    def test_cross_above(self):
        result = self.utils.cross(self.crosseddf["a"], self.crosseddf["b"])
        assert isinstance(result, Series)
        np.testing.assert_array_equal(result, self.crosseddf["crossed"])

        result = self.utils.cross(self.crosseddf["a"], self.crosseddf["b"], above=True)
        assert isinstance(result, Series)
        np.testing.assert_array_equal(result, self.crosseddf["crossed"])

    def test_cross_below(self):
        result = self.utils.cross(self.crosseddf["b"], self.crosseddf["a"], above=False)
        assert isinstance(result, Series)
        np.testing.assert_array_equal(result, self.crosseddf["crossed"])

    def test_crossover(self):
        # crossover() is cross(above=True) under a tulipy-compatible name.
        result = self.utils.crossover(self.crosseddf["a"], self.crosseddf["b"])
        assert isinstance(result, Series)
        np.testing.assert_array_equal(result, self.crosseddf["crossed"])
        assert result.name == "a_XA_b"
        assert result.category == "utility"

        expected = self.utils.cross(self.crosseddf["a"], self.crosseddf["b"], above=True)
        assert result.equals(expected)

        raw = self.utils.crossover(self.crosseddf["a"], self.crosseddf["b"], asint=False)
        assert raw.dtype == bool

        assert self.utils.crossover(None, self.crosseddf["b"]) is None

    def test_cross_value_rejects_a_non_number(self):
        # It used to reach zero() and fail as "bad operand type for abs()",
        # naming neither the function nor the argument.
        with pytest.raises(ValueError, match=r"cross_value\(\) value must be a number"):
            self.utils.cross_value(self.crosseddf["a"], "x")


# ---------------------------------------------------------------------------
# cross / cross_value decision tables: the below-cross fired on NaN bars and
# ties, the above-cross missed a touch followed by a cross.
# ---------------------------------------------------------------------------


_A = pd.Series([np.nan, np.nan, 5, 6, 4, 4, 4, 6, 5, 5, 3], name="a", dtype=float)
_B = pd.Series([5.0] * 11, name="b")


def test_cross_above_counts_touch_then_cross():
    assert pandas_ta.cross(_A, _B).tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]


def test_cross_below_ignores_nan_and_ties():
    assert pandas_ta.cross(_A, _B, above=False).tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def test_cross_value_below_ignores_warmup():
    rsi = pandas_ta.rsi(_spy().close, 14)
    got = pandas_ta.cross_value(rsi, 30, above=False)
    expected = ((rsi < 30) & (rsi.shift() >= 30)).astype(int)
    assert int(got[rsi.isna()].sum()) == 0
    assert got.tolist() == expected.tolist()


def _with_gaps(values: list, gaps: list, repeated: bool) -> pd.Series:
    """values with NaN at the drawn positions, on 0..n-1 or on an index with repeated labels."""
    index = [i // 2 for i in range(len(values))] if repeated else list(range(len(values)))
    series = pd.Series(values, index=index, dtype=float)
    series.iloc[[g % len(values) for g in gaps]] = np.nan
    return series


class TestSignalGapsProperties:
    """A missing trend or signal value makes no trade of its own.

    tsignals filled a missing trend with 0, so a gap inside an uptrend gave an
    exit and a re-entry; xsignals compared the missing bar with the thresholds,
    so a crossing that fell on the gap was lost with its whole trade. On the
    observed bars both must equal the result for the series without the gaps,
    and a gap bar carries no trade. Compared by position, so a repeated index
    label is covered too.
    """

    @staticmethod
    def _check(result: pd.DataFrame, without_gaps: pd.DataFrame, is_observed: np.ndarray) -> None:
        np.testing.assert_array_equal(result.to_numpy()[is_observed], without_gaps.to_numpy())
        # TS_Trades, by position: a non-default above/long adds its token to the name (#343)
        assert (result.iloc[:, 1].to_numpy()[~is_observed] == 0).all()

    # Gaps after the first bar: a leading NaN run means "no position yet" and,
    # unchanged, gives an entry on the first known uptrend bar.
    @given(
        st.lists(st.sampled_from([0, 1]), min_size=2, max_size=60),
        st.lists(st.integers(min_value=1, max_value=59), max_size=10),
        st.booleans(),
    )
    def test_tsignals_gaps_change_nothing_on_observed_bars(self, values, gaps, repeated):
        trend = _with_gaps(values, [max(g % len(values), 1) for g in gaps], repeated)
        is_observed = trend.notna().to_numpy()
        assume(is_observed.sum() >= 2)
        self._check(pandas_ta.tsignals(trend), pandas_ta.tsignals(trend[is_observed]), is_observed)

    @given(
        st.lists(st.floats(min_value=0.0, max_value=100.0, width=64), min_size=2, max_size=60),
        st.lists(st.integers(min_value=0, max_value=59), max_size=10),
        st.booleans(),
        st.booleans(),
        st.booleans(),
    )
    def test_xsignals_gaps_change_nothing_on_observed_bars(self, values, gaps, above, long, repeated):
        signal = _with_gaps(values, gaps, repeated)
        is_observed = signal.notna().to_numpy()
        assume(is_observed.sum() >= 2)
        kwargs = {"above": above, "long": long}
        self._check(pandas_ta.xsignals(signal, 30, 70, **kwargs), pandas_ta.xsignals(signal[is_observed], 30, 70, **kwargs), is_observed)
