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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = max(body_short_period, shadow_long_period, shadow_vs_period)
    lookback += 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_short_trail = start_idx - body_short_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vs_trail = start_idx - shadow_vs_period

    body_short_total = 0.0
    for j in range(body_short_trail, start_idx):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    shadow_long_total = 0.0
    for j in range(shadow_long_trail, start_idx):
        shadow_long_total += ca.candle_range(CandleSetting.ShadowLong, j)

    shadow_vs_total = 0.0
    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i]
            < ca.candle_average(CandleSetting.BodyShort, body_short_total, i)
            and ca.upper_shadow[i]
            > ca.candle_average(CandleSetting.ShadowLong, shadow_long_total, i)
            and ca.lower_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, shadow_vs_total, i)
            and ca.real_body_gap_down(i, i - 1)
        ):
            out[i] = 100

        # Update trailing windows
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)
        shadow_long_total += ca.candle_range(
            CandleSetting.ShadowLong, i
        ) - ca.candle_range(CandleSetting.ShadowLong, shadow_long_trail)
        shadow_vs_total += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vs_trail)
        body_short_trail += 1
        shadow_long_trail += 1
        shadow_vs_trail += 1


def cdl_invertedhammer(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Invertedhammer"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_INVERTEDHAMMER",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
