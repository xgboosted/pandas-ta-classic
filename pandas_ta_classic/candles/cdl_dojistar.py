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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = max(body_long_period, body_doji_period)
    lookback += 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    body_long_trail = start_idx + (-1) - body_long_period
    body_doji_trail = start_idx - body_doji_period

    body_long_total = 0.0
    for j in range(body_long_trail, start_idx + (-1)):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    body_doji_total = 0.0
    for j in range(body_doji_trail, start_idx):
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, j)

    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 1]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 1)
            and ca.real_body[i]
            <= ca.candle_average(CandleSetting.BodyDoji, body_doji_total, i)
            and (
                (ca.color[i - 1] == 1 and ca.real_body_gap_up(i, i - 1))
                or (ca.color[i - 1] == -1 and ca.real_body_gap_down(i, i - 1))
            )
        ):
            out[i] = -ca.color[i - 1] * 100

        # Update trailing windows
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i + (-1)
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, i) - ca.candle_range(
            CandleSetting.BodyDoji, body_doji_trail
        )
        body_long_trail += 1
        body_doji_trail += 1


def cdl_dojistar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Dojistar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DOJISTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
