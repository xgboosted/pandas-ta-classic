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


def _detect(ca, out, **kwargs):
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = equal_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    equal_trail = start_idx - 2 - equal_period

    equal_total = 0.0
    for j in range(equal_trail, start_idx - 2):
        equal_total += ca.candle_range(CandleSetting.Equal, j)

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == -1
            and ca.color[i - 1] == 1
            and ca.color[i] == -1
            and ca.low[i - 1] > ca.close[i - 2]
            and ca.close[i]
            <= ca.close[i - 2]
            + ca.candle_average(CandleSetting.Equal, equal_total, i - 2)
            and ca.close[i]
            >= ca.close[i - 2]
            - ca.candle_average(CandleSetting.Equal, equal_total, i - 2)
        ):
            out[i] = 100

        equal_total += ca.candle_range(CandleSetting.Equal, i - 2) - ca.candle_range(
            CandleSetting.Equal, equal_trail
        )
        equal_trail += 1


def cdl_sticksandwich(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Sticksandwich"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_STICKSANDWICH",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
