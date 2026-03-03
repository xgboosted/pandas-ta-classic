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
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = max(body_doji_period, shadow_long_period, near_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_doji_trail = start_idx - body_doji_period
    shadow_long_trail = start_idx - shadow_long_period
    near_trail = start_idx - near_period

    body_doji_total = 0.0
    for j in range(body_doji_trail, start_idx):
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, j)

    shadow_long_total = 0.0
    for j in range(shadow_long_trail, start_idx):
        shadow_long_total += ca.candle_range(CandleSetting.ShadowLong, j)

    near_total = 0.0
    for j in range(near_trail, start_idx):
        near_total += ca.candle_range(CandleSetting.Near, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i]
            <= ca.candle_average(CandleSetting.BodyDoji, body_doji_total, i)
            and ca.lower_shadow[i]
            > ca.candle_average(CandleSetting.ShadowLong, shadow_long_total, i)
            and ca.upper_shadow[i]
            > ca.candle_average(CandleSetting.ShadowLong, shadow_long_total, i)
            and (
                min(ca.open[i], ca.close[i])
                <= ca.low[i]
                + ca.hl_range[i] / 2.0
                + ca.candle_average(CandleSetting.Near, near_total, i)
            )
            and (
                max(ca.open[i], ca.close[i])
                >= ca.low[i]
                + ca.hl_range[i] / 2.0
                - ca.candle_average(CandleSetting.Near, near_total, i)
            )
        ):
            out[i] = 100

        # Update trailing windows
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, i) - ca.candle_range(
            CandleSetting.BodyDoji, body_doji_trail
        )
        shadow_long_total += ca.candle_range(
            CandleSetting.ShadowLong, i
        ) - ca.candle_range(CandleSetting.ShadowLong, shadow_long_trail)
        near_total += ca.candle_range(CandleSetting.Near, i) - ca.candle_range(
            CandleSetting.Near, near_trail
        )
        body_doji_trail += 1
        shadow_long_trail += 1
        near_trail += 1


def cdl_rickshawman(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Rickshawman"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_RICKSHAWMAN",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
