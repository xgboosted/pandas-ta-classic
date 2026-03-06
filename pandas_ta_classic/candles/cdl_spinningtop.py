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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = body_short_period
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]

    body_short_trail = start_idx - body_short_period
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and ca.upper_shadow[i] > ca.real_body[i]
            and ca.lower_shadow[i] > ca.real_body[i]
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_short_trail += 1


def cdl_spinningtop(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Spinningtop"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SPINNINGTOP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
