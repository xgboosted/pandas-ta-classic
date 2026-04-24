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
    start_idx = 3
    if start_idx >= len(out):
        return

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == 1
            and ca.color[i - 2] == -1
            and ca.close[i - 1] > ca.open[i - 2]
            and ca.open[i - 1] < ca.close[i - 2]
            and ca.close[i] > ca.close[i - 1]
        ) or (
            ca.color[i - 1] == -1
            and ca.color[i - 2] == 1
            and ca.open[i - 1] > ca.close[i - 2]
            and ca.close[i - 1] < ca.open[i - 2]
            and ca.close[i] < ca.close[i - 1]
        ):
            out[i] = ca.color[i - 1] * 100


def cdl_3outside(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Three Outside Up/Down"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3OUTSIDE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
