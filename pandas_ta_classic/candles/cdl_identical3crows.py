# -*- coding: utf-8 -*-
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
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

    arr_eq = ca._ranges[CandleSetting.Equal]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    svs_trail = start_idx - svs_period
    equal_trail = start_idx - equal_period

    # Seed ShadowVeryShort totals for i-2, i-1, i (indices 2, 1, 0)
    svs_total_2 = float(arr_svs[svs_trail - 2 : start_idx - 2].sum())
    svs_total_1 = float(arr_svs[svs_trail - 1 : start_idx - 1].sum())
    svs_total_0 = float(arr_svs[svs_trail:start_idx].sum())
    # Seed Equal totals for i-2, i-1 (indices 2, 1)
    equal_total_2 = float(arr_eq[equal_trail - 2 : start_idx - 2].sum())
    equal_total_1 = float(arr_eq[equal_trail - 1 : start_idx - 1].sum())
    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st black
            ca.color[i - 2] == -1
            # very short lower shadow
            and ca.lower_shadow[i - 2]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_2
            # 2nd black
            and ca.color[i - 1] == -1
            # very short lower shadow
            and ca.lower_shadow[i - 1]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_1
            # 3rd black
            and ca.color[i] == -1
            # very short lower shadow
            and ca.lower_shadow[i]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_0
            # Three declining closes
            and C[i - 2] > C[i - 1]
            and C[i - 1] > C[i]
            # 2nd opens very close to 1st close
            and O[i - 1] <= C[i - 2] + AVG_FACTOR[CandleSetting.Equal] * equal_total_2
            and O[i - 1] >= C[i - 2] - AVG_FACTOR[CandleSetting.Equal] * equal_total_2
            # 3rd opens very close to 2nd close
            and O[i] <= C[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total_1
            and O[i] >= C[i - 1] - AVG_FACTOR[CandleSetting.Equal] * equal_total_1
        ):
            out[i] = -100

        # Update ShadowVeryShort totals
        svs_total_2 += arr_svs[i - 2] - arr_svs[svs_trail - 2]
        svs_total_1 += arr_svs[i - 1] - arr_svs[svs_trail - 1]
        svs_total_0 += arr_svs[i] - arr_svs[svs_trail]
        # Update Equal totals
        equal_total_2 += arr_eq[i - 2] - arr_eq[equal_trail - 2]
        equal_total_1 += arr_eq[i - 1] - arr_eq[equal_trail - 1]

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
