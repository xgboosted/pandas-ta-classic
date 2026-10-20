"""Core candle pattern framework — translates TA-Lib's C candle macros to Python.

Underscore prefix ensures ``_build_category_dict()`` in ``_meta.py`` ignores this
file during auto-discovery.
"""

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _number
from pandas_ta_classic.utils._njit import njit

# ---------------------------------------------------------------------------
# Enums (mirror TA-Lib ta_defs.h)
# ---------------------------------------------------------------------------


class RangeType(IntEnum):
    RealBody = 0
    HighLow = 1
    Shadows = 2


class CandleSetting(IntEnum):
    BodyLong = 0
    BodyVeryLong = 1
    BodyShort = 2
    BodyDoji = 3
    ShadowLong = 4
    ShadowVeryLong = 5
    ShadowShort = 6
    ShadowVeryShort = 7
    Near = 8
    Far = 9
    Equal = 10


# ---------------------------------------------------------------------------
# Default settings  (range_type, avg_period, factor)
# From TA-Lib ta_global.c  TA_CandleDefaultSettings
# ---------------------------------------------------------------------------

CANDLE_DEFAULTS = {
    CandleSetting.BodyLong: (RangeType.RealBody, 10, 1.0),
    CandleSetting.BodyVeryLong: (RangeType.RealBody, 10, 3.0),
    CandleSetting.BodyShort: (RangeType.RealBody, 10, 1.0),
    CandleSetting.BodyDoji: (RangeType.HighLow, 10, 0.1),
    CandleSetting.ShadowLong: (RangeType.RealBody, 0, 1.0),
    CandleSetting.ShadowVeryLong: (RangeType.RealBody, 0, 2.0),
    CandleSetting.ShadowShort: (RangeType.Shadows, 10, 1.0),
    CandleSetting.ShadowVeryShort: (RangeType.HighLow, 10, 0.1),
    CandleSetting.Near: (RangeType.HighLow, 5, 0.2),
    CandleSetting.Far: (RangeType.HighLow, 5, 0.6),
    CandleSetting.Equal: (RangeType.HighLow, 5, 0.05),
}


# ---------------------------------------------------------------------------
# CandleArrays — pre-computed numpy arrays + TA-Lib macro equivalents
# ---------------------------------------------------------------------------


class CandleArrays:
    """Holds pre-computed OHLC-derived numpy arrays (TA-Lib candle-macro
    equivalents) shared by the candlestick pattern implementations."""

    __slots__ = (
        "_ranges",
        "body_high",
        "body_low",
        "close",
        "color",
        "high",
        "hl_range",
        "low",
        "lower_shadow",
        "open",
        "real_body",
        "shadow_range",
        "upper_shadow",
    )

    def __init__(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> None:
        self.open = open_
        self.high = high
        self.low = low
        self.close = close

        self.body_high = np.maximum(close, open_)
        self.body_low = np.minimum(close, open_)
        self.real_body = np.abs(close - open_)
        self.upper_shadow = high - self.body_high
        self.lower_shadow = self.body_low - low
        self.hl_range = high - low
        self.shadow_range = self.upper_shadow + self.lower_shadow
        # +1 = bullish (close >= open), -1 = bearish
        self.color = np.where(close >= open_, 1, -1)

        # Pre-computed range array for each CandleSetting (eliminates
        # per-call range-type branching).
        _rt_arrays = {
            RangeType.RealBody: self.real_body,
            RangeType.HighLow: self.hl_range,
            RangeType.Shadows: self.shadow_range,
        }
        self._ranges = {s: _rt_arrays[CANDLE_DEFAULTS[s][0]] for s in CandleSetting}


# ---------------------------------------------------------------------------
# Lookback helper
# ---------------------------------------------------------------------------


def candle_avg_period(setting: CandleSetting) -> int:
    return CANDLE_DEFAULTS[setting][1]


# ---------------------------------------------------------------------------
# Candle averages (TA_CANDLEAVERAGE with its rolling period total)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _candle_average_nb(arr, period, lag, start_idx, factor, divisor, total, out):
    # TA-Lib's TA_CANDLEAVERAGE operation for operation, ``factor * (total / period)
    # / divisor``, then ``PeriodTotal += range[i - lag] - range[trailingIdx - lag]``.
    for i in range(start_idx, len(out)):
        out[i] = factor * (total / period) / divisor
        total += arr[i - lag] - arr[i - lag - period]


def candle_average(ca: CandleArrays, setting: CandleSetting, lag: int, start_idx: int) -> np.ndarray:
    """TA-Lib's ``TA_CANDLEAVERAGE(setting, PeriodTotal, i - lag)`` for every candle ``i``.

    Element ``i`` (for ``i >= start_idx``) is the setting's factor times the mean
    range of the ``period`` candles before candle ``i - lag``, or times that
    candle's own range when the period is 0, halved for the Shadows range type.
    It uses TA-Lib's arithmetic -- the total seeded left to right from 0, then
    ``factor * (total / period) / divisor`` -- so a candle that lands exactly on
    the threshold is judged as TA-Lib judges it. Elements before *start_idx*
    are NaN; patterns never read them.

    Raises:
        ValueError: if ``start_idx < lag + period``, which would read before
            the first candle.
    """
    range_type, period, factor = CANDLE_DEFAULTS[setting]
    if start_idx < lag + period:
        raise ValueError(f"candle_average() start_idx must be >= lag + period ({lag + period}) for {setting.name}, got {start_idx}")
    return period_average(ca._ranges[setting], period, factor, 2.0 if range_type == RangeType.Shadows else 1.0, lag, start_idx)


def period_average(arr: np.ndarray, period: int, factor: float, divisor: float, lag: int, start_idx: int) -> np.ndarray:
    """``TA_CANDLEAVERAGE`` arithmetic for an arbitrary *period* and *factor*.

    The engine behind :func:`candle_average`, for patterns whose period and
    factor are parameters rather than TA-Lib settings (``cdl_doji``). The caller
    guarantees ``start_idx >= lag + period``.
    """
    out = np.empty(len(arr))
    out[:start_idx] = np.nan
    if period == 0:
        out[start_idx:] = factor * arr[start_idx - lag : len(arr) - lag] / divisor
    else:
        total = 0.0
        for value in arr[start_idx - lag - period : start_idx - lag]:
            total += value
        _candle_average_nb(arr, period, lag, start_idx, factor, divisor, total, out)
    return out


# ---------------------------------------------------------------------------
# run_pattern — top-level helper that every cdl_*.py calls
# ---------------------------------------------------------------------------


def run_pattern(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    detect_fn: Callable,
    name: str,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Validate OHLC, build CandleArrays, run *detect_fn*, finalize result.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        detect_fn: ``fn(ca: CandleArrays, out: np.ndarray, **kwargs)`` that
            fills *out* in-place with pattern signals (100 / -100 / 0).
        name: Column name, e.g. ``"CDL_HAMMER"``.
        scalar: Multiplier for output values. Default: 100.
        offset: How many periods to shift the result.
        **kwargs: Forwarded for fillna / fill_method handling.

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        A pandas Series with the pattern result, or None if validation fails.
    """
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)

    if open_ is None or high is None or low is None or close is None:
        return None

    offset = get_offset(offset)
    scalar = _number(scalar, 100, "scalar")

    arrays = [s.to_numpy(dtype=float) for s in (open_, high, low, close)]
    n = len(close)
    out = np.zeros(n, dtype=np.double)

    # A NaN bar -- a leading run from chained input, or a row resample() inserts
    # for a missing session -- would poison the running body/shadow averages for
    # the rest of the series, so every later comparison is False and the pattern
    # silently reports 0. Detect on the finite bars only, as if the NaN rows had
    # been dropped, and report 0 on the NaN rows. For a leading run this is what
    # TA-Lib does; after a mid-series NaN TA-Lib stays silent instead.
    finite = np.isfinite(np.vstack(arrays)).all(axis=0)
    if finite.all():
        detect_fn(CandleArrays(*arrays), out, **kwargs)
    elif finite.any():
        detected = np.zeros(int(finite.sum()), dtype=np.double)
        detect_fn(CandleArrays(*(a[finite] for a in arrays)), detected, **kwargs)
        out[finite] = detected

    # Scale output (TA-Lib outputs ±100; scalar lets callers adjust)
    if scalar != 100:
        mask = out != 0
        out[mask] = out[mask] / 100.0 * scalar

    result = Series(out, index=close.index)

    # Offset
    result = apply_offset(result, offset)

    # Handle fills
    result = apply_fill(result, **kwargs)

    # Name and Categorize it
    result.name = name
    result.category = "candles"

    return result
