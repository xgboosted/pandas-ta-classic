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
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 4
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing index for BodyLong setting applied to i-4
    body_long_trail = start_idx - body_long_period

    # Seed BodyLong total: sum candle_range(BodyLong, i-4)
    # for i from body_long_trail to start_idx-1
    body_long_total = 0.0
    for j in range(body_long_trail, start_idx):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j - 4)

    O = ca.open
    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st: long body
            ca.real_body[i - 4]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 4)
            # 1st, 2nd, 4th same color; 5th opposite
            and ca.color[i - 4] == ca.color[i - 3]
            and ca.color[i - 3] == ca.color[i - 1]
            and ca.color[i - 1] == -ca.color[i]
            and (
                (
                    # When 1st is black:
                    ca.color[i - 4] == -1
                    # 2nd gaps down
                    and ca.real_body_gap_down(i - 3, i - 4)
                    # 3rd has lower high and low than 2nd
                    and H[i - 2] < H[i - 3]
                    and L[i - 2] < L[i - 3]
                    # 4th has lower high and low than 3rd
                    and H[i - 1] < H[i - 2]
                    and L[i - 1] < L[i - 2]
                    # 5th closes inside the gap
                    and C[i] > O[i - 3]
                    and C[i] < C[i - 4]
                )
                or (
                    # When 1st is white:
                    ca.color[i - 4] == 1
                    # 2nd gaps up
                    and ca.real_body_gap_up(i - 3, i - 4)
                    # 3rd has higher high and low than 2nd
                    and H[i - 2] > H[i - 3]
                    and L[i - 2] > L[i - 3]
                    # 4th has higher high and low than 3rd
                    and H[i - 1] > H[i - 2]
                    and L[i - 1] > L[i - 2]
                    # 5th closes inside the gap
                    and C[i] < O[i - 3]
                    and C[i] > C[i - 4]
                )
            )
        ):
            out[i] = ca.color[i] * 100

        # Update: add current, subtract trailing (both reference i-4)
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 4
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - 4)
        body_long_trail += 1


def cdl_breakaway(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Breakaway

    A 5-candle reversal pattern. Bullish breakaway begins with a long
    bearish candle, followed by a gap-down and two more bearish candles
    with successively lower highs/lows, then a bullish candle that
    closes inside the initial gap. Bearish breakaway is the mirror.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (bullish) / -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_breakaway(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_BREAKAWAY",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
