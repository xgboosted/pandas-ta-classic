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
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(shadow_vs_period, body_long_period, equal_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_trail = start_idx - body_long_period
    equal_trail = start_idx - 1 - equal_period
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    body_long_total = float(arr_bl[body_long_trail:start_idx].sum())
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -ca.color[i]
            and ca.open[i]
            <= ca.open[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.open[i]
            >= ca.open[i - 1] - AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and (
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
            )
        ):
            out[i] = ca.color[i] * 100

        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        body_long_total += arr_bl[i] - arr_bl[body_long_trail]
        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        shadow_vs_trail += 1
        body_long_trail += 1
        equal_trail += 1


def cdl_separatinglines(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Separatinglines"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SEPARATINGLINES",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
