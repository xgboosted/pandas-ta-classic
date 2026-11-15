"""Inputs at the edge of what an indicator handles, where the contracts do not reach.

The registry-wide contracts cover an all-NaN input (``test_leading_nan_contract``),
input shorter than the window and its warning (``test_short_input_contract``), a
degenerate window and an infinite value (``test_degenerate_input``); the golden
fixtures pin the warm-up length and the valid-row count of every tracked call.
What stays here:

``test_constant_input``
    A constant price: every moving average equals the constant, a difference
    or deviation reads 0, and a constant range of 2 gives an ATR of 2.

``test_warm_up``
    The bar of the first value and no gap after it, written out per indicator:
    the golden fixtures pin the same as a valid-row count, but regenerating
    them would take a changed warm-up along.

``TestMismatchedLengths``
    Inputs of different lengths align on their shared labels; a component
    shorter than the window gives all-NaN; inputs with no label in common raise.

``test_exact_minimum_length``
    At exactly the rows a window needs, the result has exactly one value.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.assertions import assert_all_nan
from tests.contracts import inputs

_N = 60
_RNG = np.random.default_rng(seed=99)
_C = pd.Series(100.0 + np.cumsum(_RNG.standard_normal(_N)))
_H = _C + _RNG.uniform(0.2, 1.0, _N)
_L = _C - _RNG.uniform(0.2, 1.0, _N)
_V = pd.Series(_RNG.integers(500_000, 3_000_000, _N).astype(float))

_CONSTANT = 100.0
_CONST_C = pd.Series([_CONSTANT] * _N)


# ---------------------------------------------------------------------------
# Constant input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ta.ma()))
def test_constant_input_moving_average_equals_the_constant(name):
    values = ta.ma(name, _CONST_C, length=10).dropna()
    assert not values.empty, f"{name}: no value on {_N} constant bars"
    np.testing.assert_allclose(values, _CONSTANT, rtol=1e-12, err_msg=name)


@pytest.mark.parametrize("name", ["mom", "roc", "stdev"])
def test_constant_input_reads_zero(name):
    """A difference (mom, roc) or a deviation (stdev) of a constant series is 0."""
    values = getattr(ta, name)(_CONST_C, length=10, talib=False).dropna()
    assert not values.empty, f"{name}: no value on {_N} constant bars"
    assert (values.abs() < 1e-10).all(), f"{name}: {sorted(set(values))[:5]}"


def test_constant_range_gives_that_true_range_as_atr():
    atr = ta.atr(_CONST_C + 1.0, _CONST_C - 1.0, _CONST_C, length=14, talib=False)
    assert atr.first_valid_index() == 14
    np.testing.assert_allclose(atr.dropna(), 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

# (indicator, options, bar of the first value): a window of length n has its
# first value on bar n - 1; a difference (rsi, atr, roc, mom, true_range) needs
# the previous close, one bar later; a running total starts on bar 0.
WARM_UP = [
    ("sma", {"length": 10}, 9),
    ("ema", {"length": 10}, 9),
    ("wma", {"length": 10}, 9),
    ("stdev", {"length": 10}, 9),
    ("zscore", {"length": 10}, 9),
    ("cci", {"length": 14}, 13),
    ("willr", {"length": 14}, 13),
    ("rsi", {"length": 14}, 14),
    ("atr", {"length": 14}, 14),
    ("roc", {"length": 10}, 10),
    ("mom", {"length": 10}, 10),
    ("true_range", {}, 1),
    ("obv", {}, 0),
    ("ad", {}, 0),
]


@pytest.mark.parametrize(("name", "options", "first"), WARM_UP, ids=[row[0] for row in WARM_UP])
def test_warm_up_ends_on_its_bar_and_leaves_no_gap(name, options, first):
    frame = pd.DataFrame({"open": _C, "high": _H, "low": _L, "close": _C, "volume": _V})
    result = getattr(ta, name)(**inputs(name, frame), **options)
    assert result.first_valid_index() == first, f"{name}: first value on bar {result.first_valid_index()}, expected {first}"
    assert result.iloc[first:].notna().all(), f"{name}: a gap after the warm-up"


# ---------------------------------------------------------------------------
# Mismatched input lengths
# ---------------------------------------------------------------------------


class TestMismatchedLengths(TestCase):
    """Inputs of different lengths align on their shared labels; a component shorter than
    the window gives all-NaN. Inputs with no label in common raise ValueError."""

    def test_inputs_without_shared_labels_raise(self):
        dates = pd.date_range("2020-01-01", periods=len(_C), freq="D", name="date")
        close, high, low, volume = (x.set_axis(dates) for x in (_C, _H, _L, _V))
        with self.assertRaisesRegex(ValueError, r"beta\(\) inputs share no index labels: close has DatetimeIndex .* benchmark has RangeIndex"):
            ta.beta(close, _C)
        with self.assertRaisesRegex(ValueError, r"obv\(\) inputs share no index labels: close has DatetimeIndex .* volume has DatetimeIndex \(UTC\)"):
            ta.obv(close, volume.tz_localize("UTC"))
        later = high.set_axis(dates + pd.Timedelta(days=len(dates)))
        with self.assertRaisesRegex(ValueError, r"adx\(\) inputs share no index labels"):
            ta.adx(later, low, close)

    def test_partially_overlapping_inputs_still_align(self):
        dates = pd.date_range("2020-01-01", periods=len(_C), freq="D")
        close = _C.set_axis(dates)
        benchmark = _L.set_axis(dates).iloc[len(dates) // 4 :]
        result = ta.beta(close, benchmark)
        self.assertEqual(len(result), len(close))
        self.assertTrue(result.notna().any())

    def test_atr_h_too_short_returns_none(self):
        """atr with h shorter than the minimum requirement must return None."""
        with self.assertLogs("pandas_ta_classic", level="WARNING"):
            result = ta.atr(_H.iloc[:5], _L, _C, length=14, talib=False)
        assert_all_nan(self, result)

    def test_atr_l_too_short_returns_none(self):
        with self.assertLogs("pandas_ta_classic", level="WARNING"):
            result = ta.atr(_H, _L.iloc[:5], _C, length=14, talib=False)
        assert_all_nan(self, result)

    def test_atr_c_too_short_returns_none(self):
        with self.assertLogs("pandas_ta_classic", level="WARNING"):
            result = ta.atr(_H, _L, _C.iloc[:5], length=14, talib=False)
        assert_all_nan(self, result)

    def test_adx_h_too_short_returns_none(self):
        with self.assertLogs("pandas_ta_classic", level="WARNING"):
            result = ta.adx(_H.iloc[:5], _L, _C, length=14, talib=False)
        assert_all_nan(self, result)

    def test_obv_short_volume_limits_output(self):
        """
        When volume is shorter than close but still produces valid rows,
        the non-NaN count should equal len(volume).
        """
        v_short = _V.iloc[:20]
        result = ta.obv(_C, v_short, talib=False)
        self.assertIsNotNone(result)
        non_nan = int(result.notna().sum())
        self.assertEqual(
            non_nan,
            len(v_short),
            f"obv with short volume: expected {len(v_short)} non-NaN rows, got {non_nan}",
        )


# ---------------------------------------------------------------------------
# Exact minimum length
# ---------------------------------------------------------------------------

# (indicator, length, rows): rsi and roc need one more row than the window, the
# first row has no previous close to take a difference from.
EXACT_MINIMUM = [("sma", 20, 20), ("ema", 10, 10), ("stdev", 20, 20), ("rsi", 14, 15), ("roc", 10, 11)]


@pytest.mark.parametrize(("name", "length", "rows"), EXACT_MINIMUM)
def test_exact_minimum_length_gives_one_value(name, length, rows):
    result = getattr(ta, name)(_C.iloc[:rows], length=length, talib=False)
    assert int(result.notna().sum()) == 1, f"{name}({length}) on {rows} rows: {int(result.notna().sum())} values"
