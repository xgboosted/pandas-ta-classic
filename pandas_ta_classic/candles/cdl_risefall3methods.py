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
    # Lookback: max(TA_CANDLEAVGPERIOD(BodyShort), TA_CANDLEAVGPERIOD(BodyLong)) + 4
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_short_trail = start_idx - body_short_period
    body_long_trail = start_idx - body_long_period

    # [0]=BodyLong at i, [1..3]=BodyShort at i-1..i-3, [4]=BodyLong at i-4
    body_total = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Seed BodyShort totals for i-3, i-2, i-1
    j = body_short_trail
    while j < start_idx:
        body_total[3] += ca.candle_range(CandleSetting.BodyShort, j - 3)
        body_total[2] += ca.candle_range(CandleSetting.BodyShort, j - 2)
        body_total[1] += ca.candle_range(CandleSetting.BodyShort, j - 1)
        j += 1

    # Seed BodyLong totals for i-4 and i (both use BodyLong period)
    j = body_long_trail
    while j < start_idx:
        body_total[4] += ca.candle_range(CandleSetting.BodyLong, j - 4)
        body_total[0] += ca.candle_range(CandleSetting.BodyLong, j)
        j += 1

    O = ca.open
    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st long, then 3 small, 5th long
            ca.real_body[i - 4]
            > ca.candle_average(CandleSetting.BodyLong, body_total[4], i - 4)
            and ca.real_body[i - 3]
            < ca.candle_average(CandleSetting.BodyShort, body_total[3], i - 3)
            and ca.real_body[i - 2]
            < ca.candle_average(CandleSetting.BodyShort, body_total[2], i - 2)
            and ca.real_body[i - 1]
            < ca.candle_average(CandleSetting.BodyShort, body_total[1], i - 1)
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyLong, body_total[0], i)
            # white, 3 black, white  ||  black, 3 white, black
            and ca.color[i - 4] == -ca.color[i - 3]
            and ca.color[i - 3] == ca.color[i - 2]
            and ca.color[i - 2] == ca.color[i - 1]
            and ca.color[i - 1] == -ca.color[i]
            # 2nd to 4th hold within 1st: part of real body within 1st range
            and min(O[i - 3], C[i - 3]) < H[i - 4]
            and max(O[i - 3], C[i - 3]) > L[i - 4]
            and min(O[i - 2], C[i - 2]) < H[i - 4]
            and max(O[i - 2], C[i - 2]) > L[i - 4]
            and min(O[i - 1], C[i - 1]) < H[i - 4]
            and max(O[i - 1], C[i - 1]) > L[i - 4]
            # 2nd to 4th are falling (rising)
            and C[i - 2] * ca.color[i - 4] < C[i - 3] * ca.color[i - 4]
            and C[i - 1] * ca.color[i - 4] < C[i - 2] * ca.color[i - 4]
            # 5th opens above (below) the prior close
            and O[i] * ca.color[i - 4] > C[i - 1] * ca.color[i - 4]
            # 5th closes above (below) the 1st close
            and C[i] * ca.color[i - 4] > C[i - 4] * ca.color[i - 4]
        ):
            out[i] = 100 * ca.color[i - 4]

        # Update totals
        body_total[4] += ca.candle_range(
            CandleSetting.BodyLong, i - 4
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - 4)
        for k in range(3, 0, -1):
            body_total[k] += ca.candle_range(
                CandleSetting.BodyShort, i - k
            ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail - k)
        body_total[0] += ca.candle_range(CandleSetting.BodyLong, i) - ca.candle_range(
            CandleSetting.BodyLong, body_long_trail
        )
        body_short_trail += 1
        body_long_trail += 1


def cdl_risefall3methods(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Rising/Falling Three Methods

    A 5-candle continuation pattern. Rising Three Methods: long white
    candle, three small declining black candles held within the first's
    range, then a long white candle closing above the first's close.
    Falling Three Methods is the bearish mirror.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (rising) / -100 (falling) / 0, or None.

    Example:
        >>> result = cdl_risefall3methods(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_RISEFALL3METHODS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
