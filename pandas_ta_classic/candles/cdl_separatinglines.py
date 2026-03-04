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


def _detect(ca, out, **kwargs):
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(shadow_vs_period, body_long_period, equal_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_trail = start_idx - body_long_period
    equal_trail = start_idx - 1 - equal_period

    shadow_vs_total = 0.0
    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    equal_total = 0.0
    for j in range(equal_trail, start_idx - 1):
        equal_total += ca.candle_range(CandleSetting.Equal, j)

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -ca.color[i]
            and ca.open[i]
            <= ca.open[i - 1]
            + ca.candle_average(CandleSetting.Equal, equal_total, i - 1)
            and ca.open[i]
            >= ca.open[i - 1]
            - ca.candle_average(CandleSetting.Equal, equal_total, i - 1)
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i)
            and (
                (
                    ca.color[i] == 1
                    and ca.lower_shadow[i]
                    < ca.candle_average(
                        CandleSetting.ShadowVeryShort, shadow_vs_total, i
                    )
                )
                or (
                    ca.color[i] == -1
                    and ca.upper_shadow[i]
                    < ca.candle_average(
                        CandleSetting.ShadowVeryShort, shadow_vs_total, i
                    )
                )
            )
        ):
            out[i] = ca.color[i] * 100

        shadow_vs_total += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vs_trail)
        body_long_total += ca.candle_range(CandleSetting.BodyLong, i) - ca.candle_range(
            CandleSetting.BodyLong, body_long_trail
        )
        equal_total += ca.candle_range(CandleSetting.Equal, i - 1) - ca.candle_range(
            CandleSetting.Equal, equal_trail
        )
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
