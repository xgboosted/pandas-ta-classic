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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    shadow_short_period = candle_avg_period(CandleSetting.ShadowShort)
    lookback = max(body_long_period, shadow_short_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_long_trail = start_idx - body_long_period
    shadow_short_trail = start_idx - shadow_short_period

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    shadow_short_total = 0.0
    for j in range(shadow_short_trail, start_idx):
        shadow_short_total += ca.candle_range(CandleSetting.ShadowShort, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i)
            and ca.upper_shadow[i]
            < ca.candle_average(CandleSetting.ShadowShort, shadow_short_total, i)
            and ca.lower_shadow[i]
            < ca.candle_average(CandleSetting.ShadowShort, shadow_short_total, i)
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_long_total += ca.candle_range(CandleSetting.BodyLong, i) - ca.candle_range(
            CandleSetting.BodyLong, body_long_trail
        )
        shadow_short_total += ca.candle_range(
            CandleSetting.ShadowShort, i
        ) - ca.candle_range(CandleSetting.ShadowShort, shadow_short_trail)
        body_long_trail += 1
        shadow_short_trail += 1


def cdl_longline(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Longline"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_LONGLINE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
