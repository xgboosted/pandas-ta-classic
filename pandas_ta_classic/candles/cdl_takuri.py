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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    shadow_vl_period = candle_avg_period(CandleSetting.ShadowVeryLong)
    lookback = max(body_doji_period, shadow_vs_period, shadow_vl_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_doji_trail = start_idx - body_doji_period
    shadow_vs_trail = start_idx - shadow_vs_period
    shadow_vl_trail = start_idx - shadow_vl_period

    body_doji_total = 0.0
    for j in range(body_doji_trail, start_idx):
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, j)

    shadow_vs_total = 0.0
    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    shadow_vl_total = 0.0
    for j in range(shadow_vl_trail, start_idx):
        shadow_vl_total += ca.candle_range(CandleSetting.ShadowVeryLong, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i]
            <= ca.candle_average(CandleSetting.BodyDoji, body_doji_total, i)
            and ca.upper_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, shadow_vs_total, i)
            and ca.lower_shadow[i]
            > ca.candle_average(CandleSetting.ShadowVeryLong, shadow_vl_total, i)
        ):
            out[i] = 100

        # Update trailing windows
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, i) - ca.candle_range(
            CandleSetting.BodyDoji, body_doji_trail
        )
        shadow_vs_total += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vs_trail)
        shadow_vl_total += ca.candle_range(
            CandleSetting.ShadowVeryLong, i
        ) - ca.candle_range(CandleSetting.ShadowVeryLong, shadow_vl_trail)
        body_doji_trail += 1
        shadow_vs_trail += 1
        shadow_vl_trail += 1


def cdl_takuri(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Takuri"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TAKURI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
