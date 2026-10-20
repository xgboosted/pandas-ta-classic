"""Price level is not information for an additive indicator: shifting every price by ``d`` changes nothing.

Negative and zero prices are real market data, not a corner case: the WTI
front-month future settled at -37.63 USD on 2020-04-20, European day-ahead power
trades below zero for hundreds of hours a year, a back-adjusted (difference
method) continuous future drifts below zero in its early history, and spreads,
basis series and yields cross zero routinely.

An indicator built from differences, sums, extremes and ranges cannot tell
those series apart from the same series moved up by a constant. Adding ``d`` to
open, high, low and close (volume unchanged) leaves each output column either
unchanged (oscillators, ranges, volume flows, signals) or moved by ``d`` (price
levels, bands, moving averages). Three shifts are checked, one test each: one
that keeps the prices positive, one that makes them cross zero and one that
makes them all negative. Passing all three is what "supports negative prices"
means here.

What the contract leaves out, with the reason:

- ``LEVEL_IS_THE_ANSWER``: indicators that are relative to the price by
  definition -- a percent change, a log return, a ratio to the close. Their
  value depends on the level, so for a price at or below zero it has no
  meaning: the sign flips, the value explodes near zero, or a logarithm turns
  it into NaN. Listed as strict xfails, so the table cannot go stale: ``KEPT:``
  for a definition, ``TALIB:`` where the level dependence is TA-Lib's rule.
- ``LEVEL_COLUMNS``: a percent column inside an otherwise additive indicator
  (``bbands``' bandwidth, ``cpr``'s width percentage).
- ``WARMUP_BARS``: recursive filters whose state starts at ``0.0`` rather than
  at the first price, as in TA-Lib's Hilbert transform. The start-up transient
  depends on the level; after it has died out the output is exact again.
- ``NOT_A_PRICE_FUNCTION``: elementwise math (``ln``, ``sqrt``, ``acos``), binary
  operators and signal helpers, whose input is not a price series.
- ``contracts.BOUNDARY_COLUMNS``: columns whose threshold comparison rounds to
  the other side when the prices move, shared with ``test_scale_contract.py``.

What the contract does not settle: what a level-dependent indicator should do
at a price at or below zero -- raise, or return NaN and say so. Today
``mcgd`` and ``drawdown``'s ``DD_LOG`` turn NaN without a warning, and
``log_return``, ``hvol``, ``avolume`` and ``vfi`` raise only numpy's
``RuntimeWarning``. That is an open decision, so nothing here pins it; the
xfails above only say these indicators are outside the additive contract.

Indicators are discovered through ``ta.Category``.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import pandas_ta_classic as ta
from tests import contracts
from tests.contracts import BOUNDARY_COLUMNS, ELEMENTWISE, REASON_PREFIXES, inputs, params

PRICES = ["open", "high", "low", "close"]
RTOL = 1e-6

# Elementwise transforms and binary operators: their input is any series, and
# sin(x + d) has no relation to sin(x) worth stating. Signal helpers take 0/1
# signals or trends, not prices.
NOT_A_PRICE_FUNCTION = ELEMENTWISE | {"add", "sub", "mult", "div", "rolling_sum"}
NOT_A_PRICE_FUNCTION |= {"decay", "edecay", "long_run", "short_run", "tsignals", "xsignals"}

_PERCENT = "a percentage of the price; at or below zero the sign flips and near zero it explodes"
_LOG = "takes the log of a price ratio; NaN wherever the ratio is negative, i.e. where the series crosses zero"
LEVEL_IS_THE_ANSWER: dict[str, str] = {
    # rate of change and everything built on it
    "roc": _PERCENT,
    "rocp": _PERCENT,
    "rocr": "a ratio of two prices; its 1.0 neutral point assumes both are positive",
    "rocr100": "a ratio of two prices; its 100 neutral point assumes both are positive",
    "coppock": "sums two roc's: " + _PERCENT,
    "kst": "weights four roc's: " + _PERCENT,
    "trix": "the roc of a triple EMA: " + _PERCENT,
    "trixh": "the roc of a triple EMA: " + _PERCENT,
    "percent_return": _PERCENT,
    "nvi": "compounds roc on volume-down bars: " + _PERCENT,
    "pvi": "compounds roc on volume-up bars: " + _PERCENT,
    "pvt": "accumulates roc times volume: " + _PERCENT,
    # distance to an average, as a fraction of it
    "bias": "close / MA - 1: " + _PERCENT,
    "cfo": "100 * (close - TSF) / close: " + _PERCENT,
    "fosc": "100 * (close - TSF) / close: " + _PERCENT,
    "po": "100 * (fast - slow) / slow: " + _PERCENT,
    "ppo": "100 * (fast - slow) / slow: " + _PERCENT,
    "natr": "100 * ATR / close: " + _PERCENT,
    "ui": "the RMS of the percent drawdown from the rolling high: " + _PERCENT,
    "cg": "divides by the window's price sum, which is zero somewhere when the series crosses zero",
    "entropy": "treats price / sum(price) as a probability; negative prices are not a distribution",
    "beta": "regresses percent returns of close on benchmark: " + _PERCENT,
    # price times volume
    "mfi": "money flow is typical price * volume; a negative price turns inflow into outflow",
    "pvol": "price * volume: a level, not a difference",
    "vfi": "its noise cutoff is coef * close, so a negative close disables the filter",
    # threshold or encoding relative to the level
    "ht_trendmode": "TALIB: trend if |price - trendline| >= 1.5 % of the trendline (TA-Lib's rule)",
    "sarext": "TALIB: encodes a short position as -SAR (TA-Lib's format), which cannot tell it from a negative SAR",
    "mcgd": "(close / md) ** 4 in the denominator; NaN where close and md differ in sign",
    # logarithms
    "log_return": _LOG,
    "hvol": "annualised stdev of log returns: " + _LOG,
    "avolume": "uses log returns: " + _LOG,
}
LEVEL_COLUMNS: dict[str, str] = {
    "BBB_": "bandwidth: 100 * (upper - lower) / mid, " + _PERCENT,
    "ACCBL_": "the band is close * (1 - 4 * (high - low) / (high + low)), a percentage of the price",
    "ACCBU_": "the band is close * (1 + 4 * (high - low) / (high + low)), a percentage of the price",
    "CPR_WIDTH_PCT": "width as a percentage of the pivot",
    "CPR_WIDTH_CLASS": "classifies CPR_WIDTH_PCT",
    "DD_PCT": "drawdown as a fraction of the running high: " + _PERCENT,
    "DD_LOG": "log of the running high over the price: " + _LOG,
}
# Bars after which a zero-initialised filter state has died out on the data
# below; the transient before them depends on the level.
WARMUP_BARS: dict[str, int] = {
    "ebsw": 100,
    "mama": 200,
    "ht_trendline": 250,
    "ht_dcperiod": 250,
    "ht_dcphase": 250,
    "ht_phasor": 250,
    "ht_sine": 250,
}

INDICATORS = sorted(set(contracts.INDICATORS) - NOT_A_PRICE_FUNCTION)


SHIFTS = ("positive", "crossing_zero", "all_negative")


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return contracts.spy(1000)


def _shift(shift: str, df: pd.DataFrame) -> float:
    return {
        "positive": 1000.0,
        "crossing_zero": -float(df["close"].median()),
        "all_negative": -float(df["high"].max() + 100.0),
    }[shift]


def _call(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The indicator's output on ``df`` and the RuntimeWarnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = getattr(ta, name)(**inputs(name, df))
    runtime = sorted({str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)})
    return (result.to_frame() if isinstance(result, pd.Series) else result), runtime


@cache
def _base(name: str) -> pd.DataFrame:
    """The unshifted output, computed once for the three shifts; read, never changed."""
    return _call(name, contracts.spy(1000))[0]


def _same(actual: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(np.isnan(actual), np.isnan(expected)) or not np.array_equal(np.isinf(actual), np.isinf(expected)):
        return False
    finite = np.isfinite(expected)
    if not finite.any():
        return True
    atol = RTOL * max(float(np.abs(expected[finite]).max()), 1.0)
    return np.allclose(actual[finite], expected[finite], rtol=RTOL, atol=atol)


def _compared(column: str) -> bool:
    return column not in BOUNDARY_COLUMNS and not any(column.startswith(prefix) for prefix in LEVEL_COLUMNS)


def _broken(name: str, df: pd.DataFrame, shift: str, *, columns: list[str] | None = None, skip: int | None = None) -> list[str]:
    """Each column where the output under ``shift`` is neither unchanged nor moved by it.

    ``skip`` leading bars are left out, by default the indicator's ``WARMUP_BARS``.
    """
    base = _base(name)
    skip = WARMUP_BARS.get(name, 0) if skip is None else skip
    d = _shift(shift, df)
    moved_df = df.copy()
    moved_df[PRICES] += d
    moved, runtime = _call(name, moved_df)
    broken = [f"RuntimeWarning {runtime}"] if runtime else []
    assert list(moved.columns) == list(base.columns), f"{name}: columns {list(moved.columns)} after a shift, {list(base.columns)} before"
    for column in columns if columns is not None else [c for c in base.columns if _compared(c)]:
        b = base[column].to_numpy(float, na_value=np.nan)[skip:]
        m = moved[column].to_numpy(float, na_value=np.nan)[skip:]
        if _same(m, b) or _same(m, b + d):
            continue
        new_nan = int((np.isfinite(b) & ~np.isfinite(m)).sum())
        broken.append(f"{column}: " + (f"{new_nan} new NaN/inf" if new_nan else "value depends on the level"))
    return broken


_LEVEL_XFAILS = {name: reason if reason.startswith(REASON_PREFIXES) else f"KEPT: {reason}" for name, reason in LEVEL_IS_THE_ANSWER.items()}


@pytest.mark.parametrize(("name", "shift"), params([(name, shift) for shift in SHIFTS for name in INDICATORS], _LEVEL_XFAILS, key=lambda row: row[0]))
def test_output_ignores_the_price_level(name: str, shift: str, spy: pd.DataFrame) -> None:
    broken = _broken(name, spy, shift)
    assert not broken, f"{name} shifted {shift} (d={_shift(shift, spy):g}): " + "; ".join(broken)


@pytest.mark.parametrize("prefix", sorted(LEVEL_COLUMNS))
def test_level_columns_still_depend_on_the_level(prefix: str, spy: pd.DataFrame) -> None:
    """Keep ``LEVEL_COLUMNS`` honest: each exempt column must still break the contract."""
    owners = [(name, c) for name in INDICATORS if name not in LEVEL_IS_THE_ANSWER for c in _base(name).columns if c.startswith(prefix)]
    assert owners, f"no indicator produces a column starting with {prefix!r}; drop it from LEVEL_COLUMNS"
    for name, column in owners:
        still = any(_broken(name, spy, shift, columns=[column]) for shift in SHIFTS)
        assert still, f"{name} {column} now ignores the price level; drop {prefix!r} from LEVEL_COLUMNS"


@pytest.mark.parametrize("name", sorted(WARMUP_BARS))
def test_warmup_transient_depends_on_the_level(name: str, spy: pd.DataFrame) -> None:
    """Keep ``WARMUP_BARS`` honest: without the cut, the start-up transient breaks the contract."""
    assert any(_broken(name, spy, shift, skip=0) for shift in SHIFTS), f"{name} ignores the price level from the first bar; drop it from WARMUP_BARS"


def test_exemption_tables_name_registered_indicators() -> None:
    unknown = (set(LEVEL_IS_THE_ANSWER) | set(WARMUP_BARS) | NOT_A_PRICE_FUNCTION) - set(contracts.INDICATORS)
    assert not unknown, f"not in ta.Category: {sorted(unknown)}"
    assert not set(LEVEL_IS_THE_ANSWER) & set(WARMUP_BARS)


# Pure difference-and-extreme indicators, on random paths and any shift. Prices
# are whole quarters and the shift a whole number, so every shifted price and
# every price difference is exact in binary. With cent prices (0.01 is not
# exact) a tie such as adx's up-move == down-move rounds to either side after a
# shift, which is float rounding, not a dependence on the level.
_LINEAR = ["sma", "ema", "wma", "rma", "macd", "atr", "donchian", "stdev", "rsi", "stoch", "willr", "cci", "adx", "obv", "ad"]
_N = 120


@st.composite
def _ohlcv(draw) -> pd.DataFrame:
    steps = draw(st.lists(st.integers(-12, 12), min_size=_N, max_size=_N))
    spread = draw(st.lists(st.integers(0, 8), min_size=_N, max_size=_N))
    close = (400 + np.cumsum(steps)) / 4.0
    wick = np.asarray(spread) / 4.0
    volume = draw(st.lists(st.integers(1, 10**6), min_size=_N, max_size=_N))
    index = pd.date_range("2020-01-01", periods=_N, freq="D")
    return pd.DataFrame({"open": close, "high": close + wick, "low": close - wick, "close": close, "volume": np.asarray(volume, float)}, index=index)


@settings(max_examples=60, deadline=None)
@given(name=st.sampled_from(_LINEAR), df=_ohlcv(), d=st.integers(-10_000, 10_000))
def test_linear_indicators_ignore_any_shift(name: str, df: pd.DataFrame, d: int) -> None:
    base, _ = _call(name, df)
    moved_df = df.copy()
    moved_df[PRICES] += d
    moved, runtime = _call(name, moved_df)
    assert not runtime, f"{name} d={d}: {runtime}"
    for column in base.columns:
        b = base[column].to_numpy(float, na_value=np.nan)
        m = moved[column].to_numpy(float, na_value=np.nan)
        assert _same(m, b) or _same(m, b + d), f"{name} {column}: neither unchanged nor moved by d={d}"
