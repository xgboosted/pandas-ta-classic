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
    # Lookback: max(BodyShort, BodyLong) + 1
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_long_trail = start_idx - 1 - body_long_period
    body_short_trail = start_idx - body_short_period

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx - 1):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    body_short_total = 0.0
    for j in range(body_short_trail, start_idx):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    for i in range(start_idx, len(out)):
        if ca.real_body[i - 1] > ca.candle_average(
            CandleSetting.BodyLong, body_long_total, i - 1
        ) and ca.real_body[  # 1st: long
            i
        ] <= ca.candle_average(
            CandleSetting.BodyShort, body_short_total, i
        ):  # 2nd: short
            hi_i = max(ca.close[i], ca.open[i])
            lo_i = min(ca.close[i], ca.open[i])
            hi_p = max(ca.close[i - 1], ca.open[i - 1])
            lo_p = min(ca.close[i - 1], ca.open[i - 1])
            if hi_i < hi_p and lo_i > lo_p:
                out[i] = -ca.color[i - 1] * 100
            elif hi_i <= hi_p and lo_i >= lo_p:
                out[i] = -ca.color[i - 1] * 80

        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 1
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)
        body_long_trail += 1
        body_short_trail += 1


def cdl_harami(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Harami"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HARAMI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
