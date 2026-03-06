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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = max(body_long_period, body_doji_period)
    lookback += 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_bl = ca._ranges[CandleSetting.BodyLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx + (-1) - body_long_period
    body_doji_trail = start_idx - body_doji_period
    body_long_total = float(arr_bl[body_long_trail : start_idx + (-1)].sum())
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_doji_total
            and (
                (ca.color[i - 1] == 1 and body_lo[i] > body_hi[i - 1])
                or (ca.color[i - 1] == -1 and body_hi[i] < body_lo[i - 1])
            )
        ):
            out[i] = -ca.color[i - 1] * 100

        # Update trailing windows
        body_long_total += arr_bl[i + (-1)] - arr_bl[body_long_trail]
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
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
