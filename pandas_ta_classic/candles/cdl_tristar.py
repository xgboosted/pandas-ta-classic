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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = body_doji_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_trail = start_idx - 2 - body_doji_period
    body_total = float(arr_bd[body_trail : start_idx - 2].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 2] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
            and ca.real_body[i - 1] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
            and ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
        ):
            if body_lo[i - 1] > body_hi[i - 2] and body_hi[i] < max(
                ca.open[i - 1], ca.close[i - 1]
            ):
                out[i] = -100
            if (
                body_hi[i - 1] < body_lo[i - 2]
                and min(ca.open[i], ca.close[i]) > body_lo[i - 1]
            ):
                out[i] = 100

        body_total += arr_bd[i - 2] - arr_bd[body_trail]
        body_trail += 1


def cdl_tristar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Tristar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TRISTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
