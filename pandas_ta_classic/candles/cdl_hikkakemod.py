# -*- coding: utf-8 -*-
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
import numpy as np


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(1, TA_CANDLEAVGPERIOD(Near)) + 5
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = max(1, near_period) + 5
    start_idx = lookback
    if start_idx >= len(out):
        return

    H = ca.high
    L = ca.low
    C = ca.close

    # Near trailing: seeds for the Near setting applied at i-2
    # NearTrailingIdx = startIdx - 3 - near_period
    near_trail = start_idx - 3 - near_period

    # Seed Near total
    near_total = 0.0
    j = near_trail
    while j < start_idx - 3:
        near_total += ca.candle_range(CandleSetting.Near, j - 2)
        j += 1

    pattern_idx = 0
    pattern_result = 0

    # Warm-up: scan the 3 bars before start_idx
    for i in range(start_idx - 3, start_idx):
        if (
            # 2nd: lower high and higher low than 1st
            H[i - 2] < H[i - 3]
            and L[i - 2] > L[i - 3]
            # 3rd: lower high and higher low than 2nd
            and H[i - 1] < H[i - 2]
            and L[i - 1] > L[i - 2]
            and (
                (
                    # (bull) 4th: lower high and lower low
                    H[i] < H[i - 1]
                    and L[i] < L[i - 1]
                    # (bull) 2nd: close near the low
                    and C[i - 2]
                    <= L[i - 2]
                    + ca.candle_average(CandleSetting.Near, near_total, i - 2)
                )
                or (
                    # (bear) 4th: higher high and higher low
                    H[i] > H[i - 1]
                    and L[i] > L[i - 1]
                    # (bear) 2nd: close near the top
                    and C[i - 2]
                    >= H[i - 2]
                    - ca.candle_average(CandleSetting.Near, near_total, i - 2)
                )
            )
        ):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
        else:
            # Search for confirmation
            if i <= pattern_idx + 3 and (
                (pattern_result > 0 and C[i] > H[pattern_idx - 1])
                or (pattern_result < 0 and C[i] < L[pattern_idx - 1])
            ):
                pattern_idx = 0

        near_total += ca.candle_range(CandleSetting.Near, i - 2) - ca.candle_range(
            CandleSetting.Near, near_trail - 2
        )
        near_trail += 1

    # Main loop
    for i in range(start_idx, len(out)):
        if (
            # 2nd: lower high and higher low than 1st
            H[i - 2] < H[i - 3]
            and L[i - 2] > L[i - 3]
            # 3rd: lower high and higher low than 2nd
            and H[i - 1] < H[i - 2]
            and L[i - 1] > L[i - 2]
            and (
                (
                    # (bull) 4th: lower high and lower low
                    H[i] < H[i - 1]
                    and L[i] < L[i - 1]
                    # (bull) 2nd: close near the low
                    and C[i - 2]
                    <= L[i - 2]
                    + ca.candle_average(CandleSetting.Near, near_total, i - 2)
                )
                or (
                    # (bear) 4th: higher high and higher low
                    H[i] > H[i - 1]
                    and L[i] > L[i - 1]
                    # (bear) 2nd: close near the top
                    and C[i - 2]
                    >= H[i - 2]
                    - ca.candle_average(CandleSetting.Near, near_total, i - 2)
                )
            )
        ):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif i <= pattern_idx + 3 and (
            (pattern_result > 0 and C[i] > H[pattern_idx - 1])
            or (pattern_result < 0 and C[i] < L[pattern_idx - 1])
        ):
            out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
            pattern_idx = 0

        near_total += ca.candle_range(CandleSetting.Near, i - 2) - ca.candle_range(
            CandleSetting.Near, near_trail - 2
        )
        near_trail += 1


def cdl_hikkakemod(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
