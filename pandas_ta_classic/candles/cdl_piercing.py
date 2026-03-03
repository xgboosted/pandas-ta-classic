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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_long_trail = start_idx - body_long_period

    body_long_total = [0.0, 0.0]
    for j in range(body_long_trail, start_idx):
        body_long_total[1] += ca.candle_range(CandleSetting.BodyLong, j - 1)
        body_long_total[0] += ca.candle_range(CandleSetting.BodyLong, j)

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -1
            and ca.real_body[i - 1]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total[1], i - 1)
            and ca.color[i] == 1
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total[0], i)
            and ca.open[i] < ca.low[i - 1]
            and ca.close[i] < ca.open[i - 1]
            and ca.close[i] > ca.close[i - 1] + ca.real_body[i - 1] * 0.5
        ):
            out[i] = 100

        for tot_idx in range(2):
            body_long_total[tot_idx] += ca.candle_range(
                CandleSetting.BodyLong, i - tot_idx
            ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - tot_idx)
        body_long_trail += 1


def cdl_piercing(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Piercing"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_PIERCING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
