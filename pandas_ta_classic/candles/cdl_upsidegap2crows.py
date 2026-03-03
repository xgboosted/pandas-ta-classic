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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_long_trail = start_idx - 2 - body_long_period
    body_short_trail = start_idx - 1 - body_short_period

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx - 2):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    body_short_total = 0.0
    for j in range(body_short_trail, start_idx - 1):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == 1
            and ca.real_body[i - 2]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 2)
            and ca.color[i - 1] == -1
            and ca.real_body[i - 1]
            <= ca.candle_average(CandleSetting.BodyShort, body_short_total, i - 1)
            and ca.real_body_gap_up(i - 1, i - 2)
            and ca.color[i] == -1
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.close[i - 1]
            and ca.close[i] > ca.close[i - 2]
        ):
            out[i] = -100

        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 2
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i - 1
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)
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
