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
    # Lookback: max(Equal, BodyLong) + 1
    equal_period = candle_avg_period(CandleSetting.Equal)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    equal_trail = start_idx - equal_period
    body_long_trail = start_idx - body_long_period

    equal_total = 0.0
    i = equal_trail
    while i < start_idx:
        equal_total += ca.candle_range(CandleSetting.Equal, i - 1)
        i += 1

    body_long_total_1 = 0.0
    body_long_total_0 = 0.0
    i = body_long_trail
    while i < start_idx:
        body_long_total_1 += ca.candle_range(CandleSetting.BodyLong, i - 1)
        body_long_total_0 += ca.candle_range(CandleSetting.BodyLong, i)
        i += 1

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -ca.color[i]  # opposite candles
            and ca.real_body[i - 1]
            > ca.candle_average(
                CandleSetting.BodyLong, body_long_total_1, i - 1
            )  # 1st long
            and ca.real_body[i]
            > ca.candle_average(
                CandleSetting.BodyLong, body_long_total_0, i
            )  # 2nd long
            and ca.close[i]
            <= ca.close[i - 1]
            + ca.candle_average(CandleSetting.Equal, equal_total, i - 1)  # equal closes
            and ca.close[i]
            >= ca.close[i - 1]
            - ca.candle_average(CandleSetting.Equal, equal_total, i - 1)
        ):
            out[i] = ca.color[i] * 100

        equal_total += ca.candle_range(CandleSetting.Equal, i - 1) - ca.candle_range(
            CandleSetting.Equal, equal_trail - 1
        )
        for totIdx in range(1, -1, -1):
            body_long_total_1 if totIdx == 1 else body_long_total_0  # noqa
        body_long_total_1 += ca.candle_range(
            CandleSetting.BodyLong, i - 1
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - 1)
        body_long_total_0 += ca.candle_range(
            CandleSetting.BodyLong, i
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        equal_trail += 1
        body_long_trail += 1


def cdl_counterattack(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Counterattack"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_COUNTERATTACK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
