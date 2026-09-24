"""Guards for the defects found by the adversarial review of 2026-09-24.

Each test fails on the code before its fix: the expected values come from a
loop reference, TA-Lib, or a hand decision table, never from this package.

Covered:
  1. cpr previous-period levels: every bar reads the last completed calendar
     period before its own (weekly and monthly were two periods stale, then
     read their own period on the period's last day; intraday was NaN every
     Monday).
  2. sarext native path equals TA-Lib SAREXT for every parameter, not only
     the defaults (offsetonreverse, startvalue and long acceleration factors
     diverged).
  3. df.ta.adjusted naming a missing column raises instead of silently using
     'close'; an explicitly named column still works.
  4. cross / cross_value: the below-cross fired on NaN bars and ties, the
     above-cross missed a touch followed by a cross.
  5. non_zero_range on int64 prices, and bop on a flat bar.
  6. stdev forwards min_periods to variance.
  7. stochrsi(talib=True) only uses TA-Lib when it can express k and
     rsi_length, so it equals the native result.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.utils import non_zero_range
from pandas_ta_classic.utils._cpr import get_previous_period_ohlcv

_SPY = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"
needs_talib = pytest.mark.skipif(not ta.Imports["talib"], reason="TA-Lib not installed")


def _spy() -> pd.DataFrame:
    df = pd.read_csv(_SPY, index_col="date", parse_dates=True).drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# 1. cpr previous-period levels
# ---------------------------------------------------------------------------


def _ramp(index: pd.DatetimeIndex) -> pd.DataFrame:
    c = np.arange(len(index), dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=index)


def _previous_period_reference(df: pd.DataFrame, keys: list) -> np.ndarray:
    """Loop reference: first/max/min/last/sum of the previous observed period."""
    order = list(dict.fromkeys(keys))
    rows = {k: df[[x == k for x in keys]] for k in order}
    out = []
    for k in keys:
        j = order.index(k)
        if j == 0:
            out.append([np.nan] * 5)
            continue
        p = rows[order[j - 1]]
        out.append([p.open.iloc[0], p.high.max(), p.low.min(), p.close.iloc[-1], p.volume.sum()])
    return np.array(out)


_BDAYS = pd.bdate_range("2024-01-01", "2024-06-28")
_SESSIONS = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * k) for d in pd.bdate_range("2024-03-04", periods=12) for k in range(78)])
_WEEK = lambda t: t.isocalendar()[:2]
_MONTH = lambda t: (t.year, t.month)
_DAY = lambda t: t.date()


@pytest.mark.parametrize(
    "index, timeframe, key",
    [
        pytest.param(_BDAYS, "weekly", _WEEK, id="weekly-weekday-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-06-30", freq="D"), "weekly", _WEEK, id="weekly-7-day-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"), "weekly", _WEEK, id="weekly-hourly-24-7"),
        pytest.param(_BDAYS.drop(pd.bdate_range("2024-03-25", "2024-03-29")), "weekly", _WEEK, id="weekly-missing-week"),
        pytest.param(_BDAYS, "monthly", _MONTH, id="monthly-weekday-bars"),
        pytest.param(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"), "monthly", _MONTH, id="monthly-hourly-24-7"),
        pytest.param(_SESSIONS, "intraday", _DAY, id="intraday-incl-mondays"),
        pytest.param(_SESSIONS.tz_localize("America/New_York"), "intraday", _DAY, id="intraday-tz-aware"),
    ],
)
def test_cpr_previous_period_is_last_completed_period(index, timeframe, key):
    df = _ramp(index)
    got = get_previous_period_ohlcv(df, timeframe)[["prev_open", "prev_high", "prev_low", "prev_close", "prev_volume"]].to_numpy(float)
    expected = _previous_period_reference(df, [key(t) for t in df.index])
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_cpr_weekly_pivot_on_spy():
    d = _spy().iloc[-400:]
    week = d.index.to_period("W-SUN")
    prev = d.groupby(week).agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).shift(1).reindex(week).set_axis(d.index)
    expected = (prev.high + prev.low + prev.close) / 3
    got = ta.cpr(d.open, d.high, d.low, d.close, timeframe="weekly")["CPR_PIVOT"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


# ---------------------------------------------------------------------------
# 2. sarext native vs TA-Lib, every parameter
# ---------------------------------------------------------------------------

_SAREXT_CASES = {
    "defaults": {},
    "offsetonreverse": {"offsetonreverse": 0.01},
    "startvalue-long": {"startvalue": 100.0},
    "startvalue-short": {"startvalue": -150.0},
    "long-acceleration": {"accelerationinitlong": 0.01, "accelerationlong": 0.03, "accelerationmaxlong": 0.25},
    "short-acceleration": {"accelerationinitshort": 0.04, "accelerationshort": 0.01, "accelerationmaxshort": 0.1},
    "init-above-max": {"accelerationinitlong": 0.5, "accelerationmaxlong": 0.2, "accelerationshort": 0.5, "accelerationmaxshort": 0.3},
}


def _random_walk_hl(seed: int, n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"high": np.maximum(o, c) + rng.random(n), "low": np.minimum(o, c) - rng.random(n)})


@needs_talib
@pytest.mark.parametrize("kwargs", list(_SAREXT_CASES.values()), ids=list(_SAREXT_CASES))
@pytest.mark.parametrize("data", ["spy", "walk0", "walk3"])
def test_sarext_native_matches_talib(kwargs, data):
    import talib

    d = _spy() if data == "spy" else _random_walk_hl(int(data[-1]))
    got = ta.sarext(d.high, d.low, talib=False, **kwargs).to_numpy(float)
    expected = talib.SAREXT(d.high.to_numpy(float), d.low.to_numpy(float), **kwargs)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. df.ta.adjusted naming a missing column
# ---------------------------------------------------------------------------


def test_adjusted_missing_column_raises_and_explicit_column_works():
    df = _spy().iloc[-300:].copy()
    df["adj_close"] = df.close * 0.5
    df.ta.adjusted = "adj_close"
    pd.testing.assert_series_equal(df.ta.sma(), ta.sma(df.adj_close))
    subset = df[["open", "high", "low", "close", "volume"]]  # inherits _ta_adjusted through df.attrs
    for call in (subset.ta.sma, subset.ta.ichimoku, lambda: subset.ta.ma("ema")):
        with pytest.raises(KeyError, match="adj_close"):
            call()
    pd.testing.assert_series_equal(subset.ta.sma(close="close"), ta.sma(subset.close))
    subset.ta.adjusted = None
    pd.testing.assert_series_equal(subset.ta.sma(), ta.sma(subset.close))


# ---------------------------------------------------------------------------
# 4. cross / cross_value decision tables
# ---------------------------------------------------------------------------

_A = pd.Series([np.nan, np.nan, 5, 6, 4, 4, 4, 6, 5, 5, 3], name="a", dtype=float)
_B = pd.Series([5.0] * 11, name="b")


def test_cross_above_counts_touch_then_cross():
    assert ta.cross(_A, _B).tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]


def test_cross_below_ignores_nan_and_ties():
    assert ta.cross(_A, _B, above=False).tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def test_cross_value_below_ignores_warmup():
    rsi = ta.rsi(_spy().close, 14)
    got = ta.cross_value(rsi, 30, above=False)
    expected = ((rsi < 30) & (rsi.shift() >= 30)).astype(int)
    assert int(got[rsi.isna()].sum()) == 0
    assert got.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# 5. flat bars and integer prices
# ---------------------------------------------------------------------------


def test_non_zero_range_int64_matches_float():
    high = pd.Series([10, 12, 11], dtype="int64")
    low = pd.Series([10, 9, 11], dtype="int64")
    np.testing.assert_array_equal(non_zero_range(high, low).to_numpy(), non_zero_range(high.astype(float), low.astype(float)).to_numpy())


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_bop_and_ad_on_flat_bars(dtype):
    o, h, l, c, v = (pd.Series(x, dtype=dtype) for x in ([10, 10, 11], [10, 12, 11], [10, 9, 11], [10, 11, 11], [100, 200, 300]))
    assert ta.bop(o, h, l, c).tolist() == pytest.approx([0.0, 1 / 3, 0.0])  # TA-Lib BOP
    assert ta.ad(h, l, c, v).tolist() == pytest.approx([0.0, 200 / 3, 200 / 3])  # TA-Lib AD


# ---------------------------------------------------------------------------
# 6. stdev min_periods
# ---------------------------------------------------------------------------


def test_stdev_forwards_min_periods():
    x = pd.Series([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 10.0])
    got = ta.stdev(x, 5, min_periods=3)
    np.testing.assert_allclose(got, x.rolling(5, min_periods=3).std(ddof=0), equal_nan=True)
    assert np.isfinite(got.iloc[2])


# ---------------------------------------------------------------------------
# 7. stochrsi TA-Lib path honours k and rsi_length
# ---------------------------------------------------------------------------


@needs_talib
@pytest.mark.parametrize("kwargs", [{}, {"rsi_length": 7}, {"k": 1}, {"k": 1, "rsi_length": 7}], ids=["defaults", "rsi_length", "k1", "k1-rsi_length"])
def test_stochrsi_talib_path_matches_native(kwargs):
    close = _spy().close
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast, native = ta.stochrsi(close, talib=True, **kwargs), ta.stochrsi(close, talib=False, **kwargs)
    for col in range(2):
        a, b = fast.iloc[:, col].to_numpy(float), native.iloc[:, col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 5000
        np.testing.assert_allclose(a[both], b[both], rtol=0, atol=1e-9)
