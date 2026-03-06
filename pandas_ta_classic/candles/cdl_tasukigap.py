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


def _detect(ca, out, **kwargs):
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]
    body_hi = ca.body_high
    body_lo = ca.body_low

    near_trail = start_idx - near_period
    near_total = float(arr_nr[near_trail - 1 : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            body_lo[i - 1] > body_hi[i - 2]
            and ca.color[i - 1] == 1
            and ca.color[i] == -1
            and ca.open[i] < ca.close[i - 1]
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.open[i - 1]
            and ca.close[i] > body_hi[i - 2]
            and abs(ca.real_body[i - 1] - ca.real_body[i])
            < AVG_FACTOR[CandleSetting.Near] * near_total
        ) or (
            body_hi[i - 1] < body_lo[i - 2]
            and ca.color[i - 1] == -1
            and ca.color[i] == 1
            and ca.open[i] < ca.open[i - 1]
            and ca.open[i] > ca.close[i - 1]
            and ca.close[i] > ca.open[i - 1]
            and ca.close[i] < body_lo[i - 2]
            and abs(ca.real_body[i - 1] - ca.real_body[i])
            < AVG_FACTOR[CandleSetting.Near] * near_total
        ):
            out[i] = ca.color[i - 1] * 100

        near_total += arr_nr[i - 1] - arr_nr[near_trail - 1]
        near_trail += 1


def cdl_tasukigap(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Tasuki Gap"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TASUKIGAP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
