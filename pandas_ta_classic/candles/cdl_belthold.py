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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = max(body_long_period, shadow_vs_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    body_long_trail = start_idx - body_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_total = float(arr_bl[body_long_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total and (
            (
                ca.color[i] == 1
                and ca.lower_shadow[i]
                < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            )
            or (
                ca.color[i] == -1
                and ca.upper_shadow[i]
                < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            )
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_long_total += arr_bl[i] - arr_bl[body_long_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        body_long_trail += 1
        shadow_vs_trail += 1


def cdl_belthold(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Belthold"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_BELTHOLD",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
