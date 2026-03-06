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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 2 - body_long_period
    body_short_trail = start_idx - 1 - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_short_total = float(arr_bs[body_short_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == 1
            and ca.real_body[i - 2]
            > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.color[i - 1] == -1
            and ca.real_body[i - 1]
            <= AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and body_lo[i - 1] > body_hi[i - 2]
            and ca.color[i] == -1
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.close[i - 1]
            and ca.close[i] > ca.close[i - 2]
        ):
            out[i] = -100

        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i - 1] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_short_trail += 1


def cdl_upsidegap2crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Upside Gap Two Crows"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_UPSIDEGAP2CROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
