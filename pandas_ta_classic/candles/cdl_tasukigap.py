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
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    near_trail = start_idx - near_period

    near_total = 0.0
    for j in range(near_trail, start_idx):
        near_total += ca.candle_range(CandleSetting.Near, j - 1)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body_gap_up(i - 1, i - 2)
            and ca.color[i - 1] == 1
            and ca.color[i] == -1
            and ca.open[i] < ca.close[i - 1]
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.open[i - 1]
            and ca.close[i] > max(ca.close[i - 2], ca.open[i - 2])
            and abs(ca.real_body[i - 1] - ca.real_body[i])
            < ca.candle_average(CandleSetting.Near, near_total, i - 1)
        ) or (
            ca.real_body_gap_down(i - 1, i - 2)
            and ca.color[i - 1] == -1
            and ca.color[i] == 1
            and ca.open[i] < ca.open[i - 1]
            and ca.open[i] > ca.close[i - 1]
            and ca.close[i] > ca.open[i - 1]
            and ca.close[i] < min(ca.close[i - 2], ca.open[i - 2])
            and abs(ca.real_body[i - 1] - ca.real_body[i])
            < ca.candle_average(CandleSetting.Near, near_total, i - 1)
        ):
            out[i] = ca.color[i - 1] * 100

        near_total += ca.candle_range(CandleSetting.Near, i - 1) - ca.candle_range(
            CandleSetting.Near, near_trail - 1
        )
        near_trail += 1


def cdl_tasukigap(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Tasuki Gap"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TASUKIGAP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
