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
    start_idx = 2
    if start_idx >= len(out):
        return

    body_hi = ca.body_high
    body_lo = ca.body_low

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == ca.color[i - 1]
            and ca.color[i - 1] == -ca.color[i]
            and ca.open[i] < body_hi[i - 1]
            and ca.open[i] > body_lo[i - 1]
            and ca.close[i] < body_hi[i - 2]
            and ca.close[i] > body_lo[i - 2]
            and (
                (ca.color[i - 2] == 1 and body_lo[i - 1] > body_hi[i - 2])
                or (ca.color[i - 2] == -1 and body_hi[i - 1] < body_lo[i - 2])
            )
        ):
            out[i] = ca.color[i - 2] * 100


def cdl_xsidegap3methods(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Upside/Downside Gap Three Methods"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_XSIDEGAP3METHODS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
