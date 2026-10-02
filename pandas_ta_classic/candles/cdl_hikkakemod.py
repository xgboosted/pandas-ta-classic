# Candle Modified Hikkake Pattern (CDL_HIKKAKEMOD)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    CandleSetting,
    candle_average,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _hikkakemod_is_setup(H, L, C, near_2, i):
    """Check if bars at index i form a modified Hikkake setup with near condition."""
    return (
        H[i - 2] < H[i - 3]
        and L[i - 2] > L[i - 3]
        and H[i - 1] < H[i - 2]
        and L[i - 1] > L[i - 2]
        and (
            (H[i] < H[i - 1] and L[i] < L[i - 1] and C[i - 2] <= L[i - 2] + near_2[i])
            or (H[i] > H[i - 1] and L[i] > L[i - 1] and C[i - 2] >= H[i - 2] - near_2[i])
        )
    )


@njit(cache=True)
def _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
    """Check if bar i confirms a previously detected modified Hikkake pattern."""
    return i <= pattern_idx + 3 and ((pattern_result > 0 and C[i] > H[pattern_idx - 1]) or (pattern_result < 0 and C[i] < L[pattern_idx - 1]))


@njit(cache=True)
def _detect_nb(H, L, C, near_2, out, start_idx):
    pattern_idx = 0
    pattern_result = 0

    # Warm-up: scan the 3 bars before start_idx
    for i in range(start_idx - 3, start_idx):
        if _hikkakemod_is_setup(H, L, C, near_2, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
        else:
            # Search for confirmation
            if _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
                pattern_idx = 0

    # Main loop
    for i in range(start_idx, len(out)):
        if _hikkakemod_is_setup(H, L, C, near_2, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
            out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
            pattern_idx = 0


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(1, TA_CANDLEAVGPERIOD(Near)) + 5
    start_idx = max(1, candle_avg_period(CandleSetting.Near)) + 5
    if start_idx >= len(out):
        return

    # The Near average (applied to i-2) already runs over the 3 warm-up bars
    near_2 = candle_average(ca, CandleSetting.Near, 2, start_idx - 3)
    _detect_nb(ca.high, ca.low, ca.close, near_2, out, start_idx)


def cdl_hikkakemod(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Modified Hikkake

    Like the standard Hikkake but adds a requirement that the second
    candle has a close near its low (bullish) or near its high (bearish),
    and requires two nested inside bars (bar 2 inside bar 1, bar 3
    inside bar 2) before the breakout bar.

    The pattern bar outputs +/-100 and the confirmation bar outputs
    +/-200.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with pattern signals, or None.

    Example:
        >>> result = cdl_hikkakemod(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HIKKAKEMOD",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
