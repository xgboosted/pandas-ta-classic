from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from pandas_ta_classic.utils._cpr import get_previous_period_ohlcv
from tests.assertions import assert_all_nan
from tests.config import get_sample_data
from tests.contracts import spy as _spy


class TestTrend(TestCase):
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

    def test_chop(self):
        result = pandas_ta.chop(self.high, self.low, self.close, ln=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOPln_14_1_100")

    def test_cpr_virgin_detection(self):
        result = pandas_ta.cpr(self.open, self.high, self.low, self.close, virgin_cpr=True, virgin_lookforward=5)
        virgin_values = result["CPR_VIRGIN"].dropna()
        if len(virgin_values) > 0:
            self.assertTrue(virgin_values.dtype == bool)

    def test_cpr_invalid_method(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) method must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, method="invalid_method")

    def test_cpr_invalid_timeframe(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) timeframe must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, timeframe="invalid_timeframe")

    def test_cpr_invalid_levels(self):
        # An unknown choice used to fall back to the default silently.
        with self.assertRaisesRegex(ValueError, r"cpr\(\) levels must be one of"):
            pandas_ta.cpr(self.open, self.high, self.low, self.close, levels="invalid_levels")

    def test_cpr_empty_series(self):
        empty_series = Series(dtype=float)
        with self.assertLogs("pandas_ta_classic.utils._core", level="WARNING") as cm:
            result = pandas_ta.cpr(empty_series, empty_series, empty_series, empty_series)
        self.assertGreaterEqual(len(cm.output), 1)
        self.assertTrue(
            any("Series has 0 rows" in message for message in cm.output),
            f"Expected empty-series validation warning in logs: {cm.output}",
        )
        assert_all_nan(self, result)

    def test_cpr_with_nans(self):
        open_with_nan = self.open.copy()
        open_with_nan.iloc[0:5] = np.nan
        result = pandas_ta.cpr(open_with_nan, self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertIn("CPR_TC", result.columns)

    def test_decay(self):
        result = pandas_ta.decay(self.close, mode="exp")
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EXPDECAY_5")

    def test_edecay(self):
        result = pandas_ta.edecay(self.close, length=10)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EDECAY_10")

    def test_decreasing(self):
        result = pandas_ta.decreasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SDEC_3")

    def test_increasing(self):
        result = pandas_ta.increasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINC_3")

    def test_long_run(self):
        with self.assertRaisesRegex(ValueError, "long_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.long_run(None, self.open)
        with self.assertRaisesRegex(ValueError, "long_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.long_run(self.close, None)

    def test_psar_rejects_talib(self):
        """TA-Lib's SAR returns the stop alone: psar(talib=True) built PSARaf as NaN on
        every bar and PSARr as 0.0, which hid 521 reversals on SPY_D. The native stop
        equals SAR bar for bar, so psar has no talib parameter, and passing one raises."""
        for value in (True, False):
            with self.assertRaisesRegex(TypeError, "talib"):
                pandas_ta.psar(self.high, self.low, self.close, talib=value)

    def test_short_run(self):
        with self.assertRaisesRegex(ValueError, "short_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.short_run(None, self.open)
        with self.assertRaisesRegex(ValueError, "short_run\\(\\) requires 'fast' and 'slow' Series"):
            pandas_ta.short_run(self.close, None)

    def test_ttm_trend_missing_close_is_nan(self):
        # `close > avg` is False for a NaN close, which used to read as a -1
        # downtrend on the missing bar between two uptrend bars.
        high = Series(np.arange(20) + 2.0)
        low = Series(np.arange(20) * 1.0)
        close = Series(np.arange(20) + 1.0)
        close[12] = np.nan
        result = pandas_ta.ttm_trend(high, low, close)["TTM_TRND_6"]
        assert result.iloc[6:12].eq(1).all() and result.iloc[13:].eq(1).all()
        assert np.isnan(result.iloc[12])

    def test_tsignals(self):
        with self.assertRaisesRegex(ValueError, "tsignals\\(\\) requires a 'trend' Series"):
            pandas_ta.tsignals(None)

    def test_xsignals(self):
        with self.assertRaisesRegex(ValueError, "xsignals\\(\\) requires a 'signal' Series"):
            pandas_ta.xsignals(None, 70, 30)

    def test_tsignals_gap_in_the_trend_makes_no_trade(self):
        # The gap read as "no trend": an exit at bar 3 and a re-entry at bar 4.
        result = pandas_ta.tsignals(Series([0, 1, 1, np.nan, 1, 0.0]))
        self.assertEqual(result["TS_Trends"].tolist(), [0, 1, 1, 1, 1, 0])
        self.assertEqual(result["TS_Trades"].tolist(), [0, 1, 0, 0, 0, -1])

    def test_xsignals_crossing_on_a_gap_is_kept(self):
        # The up-cross through 20 fell on the missing bar, so the whole trade
        # (entry at bar 2, exit at bar 5) was lost and bars 3-4 read no trend.
        signal = Series([10, 15, np.nan, 30, 25, 15, 10.0])
        result = pandas_ta.xsignals(signal, 20, 20)
        self.assertEqual(result["TS_Trends"].tolist(), [0, 0, 0, 1, 1, 0, 0])
        self.assertEqual(result["TS_Entries"].tolist(), [0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(result["TS_Exits"].tolist(), [0, 0, 0, 0, 0, 1, 0])


# ---------------------------------------------------------------------------
# edecay is multiplicative decay floored at close (not additive)
# ---------------------------------------------------------------------------


class TestEdecayFormula(TestCase):
    """edecay uses multiplicative exp decay: result[i] = max(close[i], prev * exp(-1/n))."""

    def test_edecay_decays_after_spike(self):
        """After a spike, edecay must decay exponentially, not stay flat."""
        length = 5
        # Flat series then a spike: value should decay after the spike
        close = pd.Series([100.0] * 10 + [200.0] + [100.0] * 10)
        result = pandas_ta.edecay(close, length=length)
        self.assertIsNotNone(result)

        # Indices 11-13: decay path still > 100 floor (floor takes over at ~14)
        post_spike = result.iloc[11:14].values
        self.assertTrue(
            (post_spike > 100.0).all(),
            "edecay must stay above floor (close) right after spike",
        )
        self.assertTrue(
            (post_spike < 200.0).all(),
            "edecay must decay below spike peak",
        )

    def test_edecay_decay_rate_matches_formula(self):
        """Decay rate must match exp(-1/length) when close is below decay path."""
        import math

        length = 5
        factor = math.exp(-1.0 / length)
        # Spike then flat low floor — ensures decay path dominates
        close_vals = [10.0] * 5 + [1000.0] + [10.0] * 20
        close = pd.Series(close_vals)
        result = pandas_ta.edecay(close, length=length)

        # Two consecutive decaying bars: ratio must equal factor
        i = 6  # first bar after spike where decay path > close
        j = 7
        # Assert preconditions explicitly so a broken edecay can't silently pass
        self.assertGreater(
            result.iloc[i],
            10.0,
            f"Precondition failed: result[{i}]={result.iloc[i]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        self.assertGreater(
            result.iloc[j],
            10.0,
            f"Precondition failed: result[{j}]={result.iloc[j]:.4f} must be > 10.0 " "(decay path not yet floored at close)",
        )
        ratio = result.iloc[j] / result.iloc[i]
        self.assertAlmostEqual(
            ratio,
            factor,
            places=6,
            msg=f"edecay ratio {ratio:.6f} must equal exp(-1/{length})={factor:.6f}",
        )

    def test_edecay_never_below_close(self):
        """edecay result must always be >= close (floored at close)."""
        close = pd.Series([50.0 + i % 20 for i in range(50)], dtype=float)
        result = pandas_ta.edecay(close, length=5)
        mask = result.notna()
        self.assertTrue(
            (result[mask].values >= close[mask].values - 1e-9).all(),
            "edecay must never fall below close (floor at close)",
        )


# ---------------------------------------------------------------------------
# psar — an invalid optional close raises TypeError instead of crashing
# ---------------------------------------------------------------------------


class TestPsarCloseNoneGuard(TestCase):
    """psar() rejects an invalid optional close with TypeError, not AttributeError."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_data()
        cls.high = cls.df["high"]
        cls.low = cls.df["low"]

    def test_invalid_close_list_raises(self):
        """A plain list as close is rejected by verify_series (issue #145)."""
        bad_close = list(self.high.values)
        with self.assertRaisesRegex(TypeError, r"psar\(\) expected a pandas Series but got list"):
            pandas_ta.psar(high=self.high, low=self.low, close=bad_close)

    def test_valid_call_without_close_returns_dataframe(self):
        """Normal call without close returns a DataFrame."""
        result = pandas_ta.psar(high=self.high, low=self.low)
        self.assertIsInstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# cpr previous-period levels: every bar reads the last completed calendar
# period before its own (weekly and monthly were two periods stale, then read
# their own period on its last day; intraday was NaN every Monday).
# Found by the 2026-09-24 review; the expected values come from a loop
# reference, TA-Lib or a hand decision table, never from this package.
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
    got = pandas_ta.cpr(d.open, d.high, d.low, d.close, timeframe="weekly")["CPR_PIVOT"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


# ---------------------------------------------------------------------------
# cpr interval removal: interval= was accepted and never read; passing it
# raises TypeError instead of being swallowed by **kwargs.
# ---------------------------------------------------------------------------


def test_cpr_interval_is_rejected():
    d = _spy().iloc[-200:]
    with pytest.raises(TypeError, match="interval"):
        pandas_ta.cpr(d.open, d.high, d.low, d.close, timeframe="weekly", interval="1d")
    with pytest.raises(TypeError, match="interval"):
        d.copy().ta.cpr(timeframe="weekly", interval="1d")
