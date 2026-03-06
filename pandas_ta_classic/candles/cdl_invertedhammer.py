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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = max(body_short_period, shadow_long_period, shadow_vs_period)
    lookback += 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_short_trail = start_idx - body_short_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and ca.upper_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
            and ca.lower_shadow[i]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            and body_hi[i] < body_lo[i - 1]
        ):
            out[i] = 100

        # Update trailing windows
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
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
