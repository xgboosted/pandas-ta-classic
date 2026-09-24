"""Package output against the independent ports in tests/fixtures/reference_ports.py.

Each indicator here used to be regression-only (compared with snapshots of its
own output). Four of them were wrong against their cited definitions:
``kst`` was 100x too large, ``stc`` froze its first stochastic whenever MACD's
recent low was at or below 0, ``cksp()`` mixed the TradingView and book modes
by default, and ``ttm_trend`` included the current bar in its average (and
published -1 on warm-up bars).

``vidya`` (CMO over ``length``), ``ichimoku``, ``squeeze``, ``supertrend``
(direction), daily ``cpr`` and ``vwap`` were verified during the same review; their checks live here too.

Warm-up bars are skipped, where seed conventions differ between sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.fixtures import reference_ports as ref

WARMUP = 300


@pytest.fixture(scope="module")
def spy():
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df


def _close(got, expected, atol):
    got, expected = np.asarray(got, float), np.asarray(expected, float)
    both = np.isfinite(got) & np.isfinite(expected)
    both[:WARMUP] = False
    assert both.sum() > 4000
    np.testing.assert_allclose(got[both], expected[both], rtol=0, atol=atol)


def test_kst(spy):
    got = ta.kst(spy.close)
    line, signal = ref.kst(spy.close.to_numpy(float))
    _close(got.iloc[:, 0], line, 1e-9)
    _close(got.iloc[:, 1], signal, 1e-9)


def test_smi(spy):
    got = ta.smi(spy.close)
    for col, expected in zip(range(3), ref.smi(spy.close.to_numpy(float))):
        _close(got.iloc[:, col], expected, 1e-12)


def test_stc(spy):
    _close(ta.stc(spy.close).iloc[:, 0], ref.stc(spy.close.to_numpy(float)), 1e-9)


def test_qqe(spy):
    got = ta.qqe(spy.close)
    line, rsima = ref.qqe(spy.close.to_numpy(float))
    _close(got.iloc[:, 0], line, 1e-9)
    _close(got.filter(like="RSIMA").iloc[:, 0], rsima, 1e-9)


def test_squeeze_pro_flags(spy):
    got = ta.squeeze_pro(spy.high, spy.low, spy.close)
    # No flag before the bands exist (bar 20): it used to publish 0 there, like squeeze did.
    assert got.iloc[:20, 1:].isna().all().all()
    flags = ref.squeeze_pro_flags(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    for col, key in (("SQZPRO_ON_WIDE", "wide"), ("SQZPRO_ON_NORMAL", "normal"), ("SQZPRO_ON_NARROW", "narrow"), ("SQZPRO_OFF", "off")):
        _close(got[col], flags[key].astype(float), 0)


def test_ttm_trend(spy):
    got = ta.ttm_trend(spy.high, spy.low, spy.close).iloc[:, 0].to_numpy(float)
    expected = ref.ttm_trend(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    _close(got, expected, 0)


def test_cksp_default_is_tradingview_mode(spy):
    got = ta.cksp(spy.high, spy.low, spy.close)
    assert list(got.columns) == ["CKSPl_10_1_9", "CKSPs_10_1_9"]
    long_stop, short_stop = ref.cksp(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    _close(got.iloc[:, 0], long_stop, 1e-9)
    _close(got.iloc[:, 1], short_stop, 1e-9)


def test_ichimoku_lines(spy):
    got = ta.ichimoku(spy.high, spy.low, spy.close)
    lines = ref.ichimoku_lines(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    for col in got.columns:
        _close(got[col], lines[col.split("_")[0]], 1e-9)


def test_squeeze_flags(spy):
    got = ta.squeeze(spy.high, spy.low, spy.close)
    assert got.iloc[:20, 1:].isna().all().all()  # no flag before the bands exist
    on, off = ref.squeeze_flags(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    _close(got["SQZ_ON"], on.astype(float), 0)
    _close(got["SQZ_OFF"], off.astype(float), 0)


def test_supertrend_direction(spy):
    got = ta.supertrend(spy.high, spy.low, spy.close).iloc[:, 1]
    _close(got, ref.supertrend_direction(*(spy[c].to_numpy(float) for c in ("high", "low", "close"))), 0)


def test_cpr_daily_levels(spy):
    got = ta.cpr(spy.open, spy.high, spy.low, spy.close)
    levels = ref.cpr_daily(*(spy[c].to_numpy(float) for c in ("high", "low", "close")))
    for key, expected in levels.items():
        _close(got[f"CPR_{key}"], expected, 1e-9)


@pytest.mark.parametrize("anchor, key", [("D", lambda t: t.date()), ("W", lambda t: t.isocalendar()[:2]), ("M", lambda t: (t.year, t.month))])
def test_vwap_anchors(anchor, key):
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2024-01-29", periods=30)
    index = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * k) for d in days for k in range(78)])
    index = index.tz_localize("America/New_York")
    close = 100 + np.cumsum(rng.normal(0, 0.1, len(index)))
    high, low = close + rng.random(len(index)), close - rng.random(len(index))
    volume = rng.integers(0, 5000, len(index)).astype(float)  # includes zero-volume bars
    got = ta.vwap(*(pd.Series(x, index=index) for x in (high, low, close, volume)), anchor=anchor).to_numpy(float)
    expected = ref.vwap(high, low, close, volume, [key(t) for t in index])
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)


def test_vidya(spy):
    got = ta.vidya(spy.close).to_numpy(float)
    expected = ref.vidya(spy.close.to_numpy(float))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    _close(got, expected, 1e-9)
