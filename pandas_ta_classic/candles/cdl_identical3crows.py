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
    # Lookback: max(TA_CANDLEAVGPERIOD(ShadowVeryShort),
    #               TA_CANDLEAVGPERIOD(Equal)) + 2
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(svs_period, equal_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    svs_trail = start_idx - svs_period
    equal_trail = start_idx - equal_period

    # Seed ShadowVeryShort totals for i-2, i-1, i (indices 2, 1, 0)
    svs_total_2 = 0.0
    svs_total_1 = 0.0
    svs_total_0 = 0.0
    for j in range(svs_trail, start_idx):
        svs_total_2 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 2)
        svs_total_1 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 1)
        svs_total_0 += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    # Seed Equal totals for i-2, i-1 (indices 2, 1)
    equal_total_2 = 0.0
    equal_total_1 = 0.0
    for j in range(equal_trail, start_idx):
        equal_total_2 += ca.candle_range(CandleSetting.Equal, j - 2)
        equal_total_1 += ca.candle_range(CandleSetting.Equal, j - 1)

    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st black
            ca.color[i - 2] == -1
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
            # Three declining closes
            and C[i - 2] > C[i - 1]
            and C[i - 1] > C[i]
            # 2nd opens very close to 1st close
            and O[i - 1]
            <= C[i - 2] + ca.candle_average(CandleSetting.Equal, equal_total_2, i - 2)
            and O[i - 1]
            >= C[i - 2] - ca.candle_average(CandleSetting.Equal, equal_total_2, i - 2)
            # 3rd opens very close to 2nd close
            and O[i]
            <= C[i - 1] + ca.candle_average(CandleSetting.Equal, equal_total_1, i - 1)
            and O[i]
            >= C[i - 1] - ca.candle_average(CandleSetting.Equal, equal_total_1, i - 1)
        ):
            out[i] = -100

        # Update ShadowVeryShort totals
        svs_total_2 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 2
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 2)
        svs_total_1 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 1
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 1)
        svs_total_0 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail)
        # Update Equal totals
        equal_total_2 += ca.candle_range(CandleSetting.Equal, i - 2) - ca.candle_range(
            CandleSetting.Equal, equal_trail - 2
        )
        equal_total_1 += ca.candle_range(CandleSetting.Equal, i - 1) - ca.candle_range(
            CandleSetting.Equal, equal_trail - 1
        )

        svs_trail += 1
        equal_trail += 1


def cdl_identical3crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Identical Three Crows

    Three consecutive declining black (bearish) candlesticks, each with
    very short lower shadows. Each candle after the first opens at or
    very close to the prior candle's close (the "identical" open
    distinguishes this from regular Three Black Crows).

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
        >>> result = cdl_identical3crows(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_IDENTICAL3CROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
