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
    penetration = kwargs.get("penetration", 0.5)
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 1
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]

    body_long_trail = start_idx - body_long_period

    body_long_total = 0.0
    i = body_long_trail
    while i < start_idx:
        body_long_total += arr_bl[i - 1]
        i += 1

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == 1  # 1st: white
            and ca.real_body[i - 1]
            > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total  # long
            and ca.color[i] == -1  # 2nd: black
            and ca.open[i] > ca.high[i - 1]  # open above prior high
            and ca.close[i] > ca.open[i - 1]  # close within prior body
            and ca.close[i] < ca.close[i - 1] - ca.real_body[i - 1] * penetration
        ):
            out[i] = -100

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail - 1]
        body_long_trail += 1


def cdl_darkcloudcover(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: Optional[float] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Dark Cloud Cover"""
    if penetration is None:
        penetration = 0.5
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DARKCLOUDCOVER",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
