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
    # Lookback: TA_CANDLEAVGPERIOD(Near) + 3
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 3
    start_idx = lookback
    if start_idx >= len(out):
        return

    near_trail = start_idx - near_period

    # Seed Near totals for i-3 and i-2 (indices 3, 2)
    near_total_3 = 0.0
    near_total_2 = 0.0
    for j in range(near_trail, start_idx):
        near_total_3 += ca.candle_range(CandleSetting.Near, j - 3)
        near_total_2 += ca.candle_range(CandleSetting.Near, j - 2)

    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # Three candles with same color
            ca.color[i - 3] == ca.color[i - 2]
            and ca.color[i - 2] == ca.color[i - 1]
            # 4th opposite color
            and ca.color[i] == -ca.color[i - 1]
            # 2nd opens within/near 1st real body
            and O[i - 2]
            >= min(O[i - 3], C[i - 3])
            - ca.candle_average(CandleSetting.Near, near_total_3, i - 3)
            and O[i - 2]
            <= max(O[i - 3], C[i - 3])
            + ca.candle_average(CandleSetting.Near, near_total_3, i - 3)
            # 3rd opens within/near 2nd real body
            and O[i - 1]
            >= min(O[i - 2], C[i - 2])
            - ca.candle_average(CandleSetting.Near, near_total_2, i - 2)
            and O[i - 1]
            <= max(O[i - 2], C[i - 2])
            + ca.candle_average(CandleSetting.Near, near_total_2, i - 2)
            and (
                (
                    # If three white
                    ca.color[i - 1] == 1
                    # Consecutive higher closes
                    and C[i - 1] > C[i - 2]
                    and C[i - 2] > C[i - 3]
                    # 4th opens above prior close
                    and O[i] > C[i - 1]
                    # 4th closes below 1st open
                    and C[i] < O[i - 3]
                )
                or (
                    # If three black
                    ca.color[i - 1] == -1
                    # Consecutive lower closes
                    and C[i - 1] < C[i - 2]
                    and C[i - 2] < C[i - 3]
                    # 4th opens below prior close
                    and O[i] < C[i - 1]
                    # 4th closes above 1st open
                    and C[i] > O[i - 3]
                )
            )
        ):
            out[i] = ca.color[i - 1] * 100

        # Update totals: add current range, subtract trailing range
        near_total_3 += ca.candle_range(CandleSetting.Near, i - 3) - ca.candle_range(
            CandleSetting.Near, near_trail - 3
        )
        near_total_2 += ca.candle_range(CandleSetting.Near, i - 2) - ca.candle_range(
            CandleSetting.Near, near_trail - 2
        )
        near_trail += 1


def cdl_3linestrike(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Three-Line Strike

    Three same-colored candles with consecutively higher (white) or lower
    (black) closes, each opening within or near the prior real body. The
    fourth candle is the opposite color and engulfs the entire 3-candle
    move (opening beyond the third's close, closing beyond the first's
    open).

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
        >>> result = cdl_3linestrike(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3LINESTRIKE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
