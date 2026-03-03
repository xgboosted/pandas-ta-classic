# -*- coding: utf-8 -*-
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    run_pattern,
)
import numpy as np


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: 2 (no candle settings needed)
    start_idx = 2
    if start_idx >= len(out):
        return

    for i in range(start_idx, len(out)):
        if (
            ca.color[i] == 1
            and ca.color[i - 1] == -1  # white engulfs black
            and (
                (ca.close[i] >= ca.open[i - 1] and ca.open[i] < ca.close[i - 1])
                or (ca.close[i] > ca.open[i - 1] and ca.open[i] <= ca.close[i - 1])
            )
        ) or (
            ca.color[i] == -1
            and ca.color[i - 1] == 1  # black engulfs white
            and (
                (ca.open[i] >= ca.close[i - 1] and ca.close[i] < ca.open[i - 1])
                or (ca.open[i] > ca.close[i - 1] and ca.close[i] <= ca.open[i - 1])
            )
        ):
            if ca.open[i] != ca.close[i - 1] and ca.close[i] != ca.open[i - 1]:
                out[i] = ca.color[i] * 100
            else:
                out[i] = ca.color[i] * 80


def cdl_engulfing(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Engulfing"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ENGULFING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
