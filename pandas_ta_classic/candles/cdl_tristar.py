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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = body_doji_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_trail = start_idx - 2 - body_doji_period

    body_total = 0.0
    for j in range(body_trail, start_idx - 2):
        body_total += ca.candle_range(CandleSetting.BodyDoji, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 2]
            <= ca.candle_average(CandleSetting.BodyDoji, body_total, i - 2)
            and ca.real_body[i - 1]
            <= ca.candle_average(CandleSetting.BodyDoji, body_total, i - 2)
            and ca.real_body[i]
            <= ca.candle_average(CandleSetting.BodyDoji, body_total, i - 2)
        ):
            if ca.real_body_gap_up(i - 1, i - 2) and max(ca.open[i], ca.close[i]) < max(
                ca.open[i - 1], ca.close[i - 1]
            ):
                out[i] = -100
            if ca.real_body_gap_down(i - 1, i - 2) and min(
                ca.open[i], ca.close[i]
            ) > min(ca.open[i - 1], ca.close[i - 1]):
                out[i] = 100

        body_total += ca.candle_range(CandleSetting.BodyDoji, i - 2) - ca.candle_range(
            CandleSetting.BodyDoji, body_trail
        )
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
