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
    body_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)  # 0
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    near_period = candle_avg_period(CandleSetting.Near)

    # Lookback: max of all periods + 1 (extra bar for i-1 reference)
    lookback = max(body_period, shadow_long_period, shadow_vs_period, near_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices
    body_trail = start_idx - body_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    near_trail = start_idx - 1 - near_period  # Near uses i-1

    # Seed totals
    body_total = 0.0
    for j in range(body_trail, start_idx):
        body_total += ca.candle_range(CandleSetting.BodyShort, j)

    shadow_long_total = 0.0
    for j in range(shadow_long_trail, start_idx):
        shadow_long_total += ca.candle_range(CandleSetting.ShadowLong, j)

    shadow_vs_total = 0.0
    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    near_total = 0.0
    for j in range(near_trail, start_idx - 1):
        near_total += ca.candle_range(CandleSetting.Near, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < ca.candle_average(CandleSetting.BodyShort, body_total, i)
            and ca.lower_shadow[i]
            > ca.candle_average(CandleSetting.ShadowLong, shadow_long_total, i)
            and ca.upper_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, shadow_vs_total, i)
            and (
                min(ca.close[i], ca.open[i])
                <= ca.low[i - 1]
                + ca.candle_average(CandleSetting.Near, near_total, i - 1)
            )
        ):
            out[i] = 100

        # Update trailing windows AFTER pattern check
        body_total += ca.candle_range(CandleSetting.BodyShort, i) - ca.candle_range(
            CandleSetting.BodyShort, body_trail
        )
        shadow_long_total += ca.candle_range(
            CandleSetting.ShadowLong, i
        ) - ca.candle_range(CandleSetting.ShadowLong, shadow_long_trail)
        shadow_vs_total += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vs_trail)
        near_total += ca.candle_range(CandleSetting.Near, i - 1) - ca.candle_range(
            CandleSetting.Near, near_trail
        )

        body_trail += 1
        shadow_long_trail += 1
        shadow_vs_trail += 1
        near_trail += 1


def cdl_hammer(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Hammer"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HAMMER",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
