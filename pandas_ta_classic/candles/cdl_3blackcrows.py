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
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 3
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = svs_period + 3
    start_idx = lookback
    if start_idx >= len(out):
        return

    svs_trail = start_idx - svs_period

    # Seed ShadowVeryShort totals for i-2, i-1, i (indices 2, 1, 0)
    svs_total_2 = 0.0
    svs_total_1 = 0.0
    svs_total_0 = 0.0
    for j in range(svs_trail, start_idx):
        svs_total_2 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 2)
        svs_total_1 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 1)
        svs_total_0 += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    O = ca.open
    H = ca.high
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # Prior candle (i-3) is white
            ca.color[i - 3] == 1
            # 1st black
            and ca.color[i - 2] == -1
            # very short lower shadow
            and ca.lower_shadow[i - 2]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_2, i - 2)
            # 2nd black
            and ca.color[i - 1] == -1
            # very short lower shadow
            and ca.lower_shadow[i - 1]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_1, i - 1)
            # 3rd black
            and ca.color[i] == -1
            # very short lower shadow
            and ca.lower_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_0, i)
            # 2nd black opens within 1st black's real body
            and O[i - 1] < O[i - 2]
            and O[i - 1] > C[i - 2]
            # 3rd black opens within 2nd black's real body
            and O[i] < O[i - 1]
            and O[i] > C[i - 1]
            # 1st black closes under prior candle's high
            and H[i - 3] > C[i - 2]
            # Three declining closes
            and C[i - 2] > C[i - 1]
            and C[i - 1] > C[i]
        ):
            out[i] = -100

        # Update totals: add current range, subtract trailing range
        svs_total_2 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 2
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 2)
        svs_total_1 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 1
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 1)
        svs_total_0 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail)
        svs_trail += 1


def cdl_3blackcrows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Three Black Crows

    Three consecutive declining black (bearish) candlesticks, each with
    very short lower shadows. Each candle after the first opens within
    the prior candle's real body. The first candle's close is below the
    preceding white candle's high.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_3blackcrows(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3BLACKCROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
