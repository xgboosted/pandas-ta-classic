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
    # Lookback: max(BodyShort, BodyLong) + 1
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 1 - body_long_period
    body_short_trail = start_idx - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.real_body[i]
            <= AVG_FACTOR[CandleSetting.BodyShort] * body_short_total  # 1st: long
        ):  # 2nd: short
            hi_i = body_hi[i]
            lo_i = body_lo[i]
            hi_p = body_hi[i - 1]
            lo_p = body_lo[i - 1]
            if hi_i < hi_p and lo_i > lo_p:
                out[i] = -ca.color[i - 1] * 100
            elif hi_i <= hi_p and lo_i >= lo_p:
                out[i] = -ca.color[i - 1] * 80

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
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
