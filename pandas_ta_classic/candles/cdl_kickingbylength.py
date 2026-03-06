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
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(shadow_vs_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    hi = ca.high
    lo = ca.low

    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_trail = start_idx - body_long_period

    shadow_vs_total = [0.0, 0.0]
    body_long_total = [0.0, 0.0]

    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total[1] += arr_svs[j - 1]
        shadow_vs_total[0] += arr_svs[j]

    for j in range(body_long_trail, start_idx):
        body_long_total[1] += arr_bl[j - 1]
        body_long_total[0] += arr_bl[j]

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -ca.color[i]
            and ca.real_body[i - 1]
            > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total[1]
            and ca.upper_shadow[i - 1]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total[1]
            and ca.lower_shadow[i - 1]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total[1]
            and ca.real_body[i]
            > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total[0]
            and ca.upper_shadow[i]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total[0]
            and ca.lower_shadow[i]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total[0]
            and (
                (ca.color[i - 1] == -1 and lo[i] > hi[i - 1])
                or (ca.color[i - 1] == 1 and hi[i] < lo[i - 1])
            )
        ):
            out[i] = (
                ca.color[i if ca.real_body[i] > ca.real_body[i - 1] else i - 1] * 100
            )

        for tot_idx in range(2):
            body_long_total[tot_idx] += (
                arr_bl[i - tot_idx] - arr_bl[body_long_trail - tot_idx]
            )
            shadow_vs_total[tot_idx] += (
                arr_svs[i - tot_idx] - arr_svs[shadow_vs_trail - tot_idx]
            )
        body_long_trail += 1
        shadow_vs_trail += 1


def cdl_kickingbylength(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Kickingbylength"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_KICKINGBYLENGTH",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
