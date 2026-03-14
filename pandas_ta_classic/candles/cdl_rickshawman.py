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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = max(body_doji_period, shadow_long_period, near_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_nr = ca._ranges[CandleSetting.Near]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_doji_trail = start_idx - body_doji_period
    shadow_long_trail = start_idx - shadow_long_period
    near_trail = start_idx - near_period
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())
    near_total = float(arr_nr[near_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_doji_total
            and ca.lower_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
            and ca.upper_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
            and (
                body_lo[i]
                <= ca.low[i]
                + ca.hl_range[i] / 2.0
                + AVG_FACTOR[CandleSetting.Near] * near_total
            )
            and (
                body_hi[i]
                >= ca.low[i]
                + ca.hl_range[i] / 2.0
                - AVG_FACTOR[CandleSetting.Near] * near_total
            )
        ):
            out[i] = 100

        # Update trailing windows
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        near_total += arr_nr[i] - arr_nr[near_trail]
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
