"""What an indicator's values must satisfy on SPY_D, whatever their exact numbers.

``tests/snapshots`` compares the tracked calls with stored values (the last
value and valid count of each column, and 24 checkpoints); these are the
properties that hold by definition, for every registered indicator where they
apply: each call has a value on the last bar apart from the gaps listed (its
default call, each boolean option flipped, each optional price input left
out), an oscillator stays in its range, a deviation is never negative, a flag
is 0 or 1, ``hl2`` lies between the low and the high, and a dispatcher returns
what the function it dispatches to returns.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, candidate_calls, inputs, spy

INF = float("inf")

# (indicator, options, column, low, high): every value lies in [low, high].
BOUNDED = [
    ("adx", {}, "ADX_14", 0.0, 100.0),
    ("aroon", {}, "AROOND_14", 0.0, 100.0),
    ("aroon", {}, "AROONU_14", 0.0, 100.0),
    ("chop", {}, None, 0.0, 100.0),
    ("cti", {}, None, -1.0, 1.0),
    ("er", {}, None, 0.0, 1.0),
    # floating-point rounding in the Laguerre filter (observed max 100.0000000000001)
    ("lrsi", {}, None, 0.0, 100.0 + 1e-9),
    ("psl", {}, None, 0.0, 100.0),
    ("rsx", {}, None, 0.0, 100.0),
    # deviations and distances
    ("entropy", {}, None, 0.0, INF),
    ("mad", {}, None, 0.0, INF),
    ("pdist", {}, None, 0.0, INF),
    ("pvol", {}, None, 0.0, INF),
    ("ui", {}, None, 0.0, INF),
    ("variance", {}, None, 0.0, INF),
]

# Indicators whose every value is a 0/1 flag.
BINARY = ["decreasing", "increasing", "long_run", "short_run"]


# Columns without a value on SPY_D's last bar, by design; None stands for every column.
_ONE_SIDE = "KEPT: one of the long/short columns holds the value on each bar; SPY_D ends in an uptrend, so the short side is NaN"
LAST_BAR_GAPS = {
    ("acos", None): "KEPT: close is outside acos's domain [-1, 1] on every bar",
    ("asin", None): "KEPT: close is outside asin's domain [-1, 1] on every bar",
    ("dpo", "DPO_20"): "KEPT: the centered line reads bars after the last one",
    ("ichimoku", "ICS_26"): "KEPT: the chikou span reads the close 26 bars ahead, which the last 26 bars do not have",
    ("hilo", "HILOs_13_21"): _ONE_SIDE,
    ("psar", "PSARs_0.02_0.2"): _ONE_SIDE,
    ("qqe", "QQEs_14_5_4.236"): _ONE_SIDE,
    ("supertrend", "SUPERTs_7_3.0"): _ONE_SIDE,
    ("td_seq", "TD_SEQ_DNa"): "KEPT: the down count has a value only during a down sequence; none runs on SPY_D's last bar",
}
# append_span adds 26 future rows past the last bar, on which only the spans have values.
_SPAN_ROWS = {"ITS_9", "IKS_26"}


def _calls() -> list:
    """Every registered indicator's default call, each boolean option flipped and each optional price input left out."""
    calls = []
    for name in INDICATORS:
        calls.append(pytest.param(name, {}, None, id=name))
        calls += [pytest.param(name, call, None, id="-".join([name, *(f"{k}-{v}" for k, v in call.items())])) for call in candidate_calls(name)[1:]]
        signature = inspect.signature(getattr(ta, name)).parameters
        optional = [p for p in ("open_", "high", "low", "close", "volume") if p in signature and signature[p].default is not inspect.Parameter.empty]
        calls += [pytest.param(name, {}, p, id=f"{name}-without-{p}") for p in optional]
    return calls


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return spy()


@pytest.mark.parametrize(("name", "options", "dropped"), _calls())
def test_every_column_has_a_value_on_the_last_bar(name, options, dropped, frame):
    kwargs = {k: v for k, v in inputs(name, frame).items() if k != dropped}
    result = getattr(ta, name)(**kwargs, **options)
    if result is None:
        assert dropped, f"{name}({options}) returned None"
        pytest.skip(f"{name} needs {dropped}: without it there is no input, and the result is None")
    result = result.to_frame() if isinstance(result, pd.Series) else result
    gaps = {column for (indicator, column) in LAST_BAR_GAPS if indicator == name}
    if name == "ichimoku" and options.get("append_span"):
        gaps |= _SPAN_ROWS
    missing = [c for c in result.columns if None not in gaps and c not in gaps and not np.isfinite(pd.to_numeric(result[c], errors="coerce").iloc[-1])]
    assert not missing, f"{name}({options}{', without ' + dropped if dropped else ''}) has no value on the last bar in {missing}"


def test_every_gap_is_still_a_gap(frame):
    """A listed gap that has a value again is removed from LAST_BAR_GAPS."""
    stale = []
    for name, column in LAST_BAR_GAPS:
        result = getattr(ta, name)(**inputs(name, frame))
        result = result.to_frame() if isinstance(result, pd.Series) else result
        columns = result.columns if column is None else [column]
        if all(np.isfinite(pd.to_numeric(result[c], errors="coerce").iloc[-1]) for c in columns):
            stale.append((name, column))
    assert not stale, f"no longer a gap: {stale}"


def _column(result, column: str | None) -> pd.Series:
    return result if column is None else result[column]


@pytest.mark.parametrize(("indicator", "options", "column", "low", "high"), [pytest.param(*row, id="-".join(filter(None, (row[0], row[2])))) for row in BOUNDED])
def test_values_stay_in_their_range(indicator, options, column, low, high, frame):
    values = _column(getattr(ta, indicator)(**inputs(indicator, frame), **options), column).dropna()
    assert not values.empty, f"{indicator}: no value on SPY_D"
    outside = values[(values < low) | (values > high)]
    assert outside.empty, f"{indicator} {column or ''}: {len(outside)} values outside [{low}, {high}], min {values.min()}, max {values.max()}"


@pytest.mark.parametrize("indicator", BINARY)
def test_flags_are_zero_or_one(indicator, frame):
    values = getattr(ta, indicator)(**inputs(indicator, frame)).dropna()
    assert not values.empty, f"{indicator}: no value on SPY_D"
    assert set(values.unique()) <= {0.0, 1.0}, f"{indicator}: values {sorted(set(values.unique()))}"


def test_hl2_lies_between_low_and_high(frame):
    hl2 = ta.hl2(frame.high, frame.low)
    assert ((hl2 >= frame.low) & (hl2 <= frame.high)).all()


def test_heikin_ashi_high_is_not_below_its_low(frame):
    ha = ta.ha(frame.open, frame.high, frame.low, frame.close)
    assert (ha["HA_high"] >= ha["HA_low"]).all()


def test_cumulative_log_return_is_the_running_sum_of_log_returns(frame):
    np.testing.assert_allclose(ta.log_return(frame.close, cumulative=True), ta.log_return(frame.close).fillna(0).cumsum(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", sorted(ta.ma()))
def test_ma_returns_what_the_moving_average_returns(name, frame):
    # 7 is no moving average's default, so a dispatcher that dropped the options would show
    pd.testing.assert_series_equal(ta.ma(name, frame.close, length=7), getattr(ta, name)(frame.close, length=7))
