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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    equal_trail = start_idx - 1 - equal_period
    body_long_trail = start_idx - 1 - body_long_period

    equal_total = 0.0
    for j in range(equal_trail, start_idx - 1):
        equal_total += ca.candle_range(CandleSetting.Equal, j)

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx - 1):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -1
            and ca.real_body[i - 1]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 1)
            and ca.color[i] == 1
            and ca.open[i] < ca.low[i - 1]
            and ca.close[i]
            > ca.close[i - 1]
            + ca.candle_average(CandleSetting.Equal, equal_total, i - 1)
            and ca.close[i] <= ca.open[i - 1] - ca.real_body[i - 1] * 0.5
        ):
            out[i] = -100

        equal_total += ca.candle_range(CandleSetting.Equal, i - 1) - ca.candle_range(
            CandleSetting.Equal, equal_trail
        )
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 1
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        equal_trail += 1
        body_long_trail += 1


def cdl_thrusting(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Thrusting"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_THRUSTING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
