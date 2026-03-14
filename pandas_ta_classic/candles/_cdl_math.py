# -*- coding: utf-8 -*-
"""Core candle pattern framework — translates TA-Lib's C candle macros to Python.

Underscore prefix ensures ``_build_category_dict()`` in ``_meta.py`` ignores this
file during auto-discovery.
"""
from typing import Any, Callable, Optional

from enum import IntEnum

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import get_offset, verify_series


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
# Pre-computed average parameters (module-level for direct access in _detect)
# ---------------------------------------------------------------------------

AVG_PERIOD = {s: CANDLE_DEFAULTS[s][1] for s in CandleSetting}

AVG_FACTOR = {}
for _s in CandleSetting:
    _rt, _ap, _f = CANDLE_DEFAULTS[_s]
    _d = 2.0 if _rt == RangeType.Shadows else 1.0
    AVG_FACTOR[_s] = _f / (_ap * _d) if _ap != 0 else _f / _d
del _s, _rt, _ap, _f, _d


# ---------------------------------------------------------------------------
# CandleArrays — pre-computed numpy arrays + TA-Lib macro equivalents
# ---------------------------------------------------------------------------


class CandleArrays:
    """Holds pre-computed OHLC-derived numpy arrays and provides TA-Lib
    macro equivalents (``candle_range``, ``candle_average``, etc.)."""

    __slots__ = (
        "open",
        "high",
        "low",
        "close",
        "real_body",
        "upper_shadow",
        "lower_shadow",
        "hl_range",
        "shadow_range",
        "body_high",
        "body_low",
        "color",
        "_ranges",
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
        # per-call branching in candle_range).
        _rt_arrays = {
            RangeType.RealBody: self.real_body,
            RangeType.HighLow: self.hl_range,
            RangeType.Shadows: self.shadow_range,
        }
        self._ranges = {s: _rt_arrays[CANDLE_DEFAULTS[s][0]] for s in CandleSetting}

    # -- TA-Lib macro: TA_CANDLERANGE --
    def candle_range(self, setting: CandleSetting, i: int) -> float:
        return self._ranges[setting][i]

    # -- TA-Lib macro: TA_CANDLEAVERAGE --
    def candle_average(
        self, setting: CandleSetting, period_total: float, i: int
    ) -> float:
        """Exact replica of TA-Lib's TA_CANDLEAVERAGE macro.

        Formula: ``factor * (sum/period) / (2 if Shadows else 1)``
        When ``avgPeriod == 0``, uses ``candle_range(setting, i)`` instead of
        ``sum/period``.
        """
        ap = AVG_PERIOD[setting]
        if ap != 0:
            return AVG_FACTOR[setting] * period_total
        return AVG_FACTOR[setting] * self._ranges[setting][i]

    # -- TA-Lib macro: TA_REALBODYGAPUP --
    def real_body_gap_up(self, i2: int, i1: int) -> bool:
        return self.body_low[i2] > self.body_high[i1]

    # -- TA-Lib macro: TA_REALBODYGAPDOWN --
    def real_body_gap_down(self, i2: int, i1: int) -> bool:
        return self.body_high[i2] < self.body_low[i1]

    # -- TA-Lib macro: TA_CANDLEGAPUP --
    def candle_gap_up(self, i2: int, i1: int) -> bool:
        return self.low[i2] > self.high[i1]

    # -- TA-Lib macro: TA_CANDLEGAPDOWN --
    def candle_gap_down(self, i2: int, i1: int) -> bool:
        return self.high[i2] < self.low[i1]


# ---------------------------------------------------------------------------
# Lookback helper
# ---------------------------------------------------------------------------


def candle_avg_period(setting: CandleSetting) -> int:
    return CANDLE_DEFAULTS[setting][1]


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
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
    scalar = float(scalar) if scalar else 100

    ca = CandleArrays(
        open_.to_numpy(dtype=float),
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        close.to_numpy(dtype=float),
    )

    n = len(close)
    out = np.zeros(n, dtype=np.double)

    detect_fn(ca, out, **kwargs)

    # Scale output (TA-Lib outputs ±100; scalar lets callers adjust)
    if scalar != 100:
        mask = out != 0
        out[mask] = out[mask] / 100.0 * scalar

    result = Series(out, index=close.index)

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            result.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            result.bfill(inplace=True)

    # Name and Categorize it
    result.name = name
    result.category = "candles"

    return result
