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
    equal_period = candle_avg_period(CandleSetting.Equal)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 1 - equal_period
    body_long_trail = start_idx - 1 - body_long_period
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -1
            and ca.real_body[i - 1]
            > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.color[i] == 1
            and ca.open[i] < ca.low[i - 1]
            and ca.close[i]
            <= ca.low[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.close[i]
            >= ca.low[i - 1] - AVG_FACTOR[CandleSetting.Equal] * equal_total
        ):
            out[i] = -100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        equal_trail += 1
        body_long_trail += 1


def cdl_onneck(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Onneck"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ONNECK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
