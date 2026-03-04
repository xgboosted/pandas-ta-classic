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
    # Lookback: max(Near, Equal) + 2
    near_period = candle_avg_period(CandleSetting.Near)
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(near_period, equal_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    near_trail = start_idx - near_period
    equal_trail = start_idx - equal_period

    near_total = 0.0
    i = near_trail
    while i < start_idx:
        near_total += ca.candle_range(CandleSetting.Near, i - 1)
        i += 1

    equal_total = 0.0
    i = equal_trail
    while i < start_idx:
        equal_total += ca.candle_range(CandleSetting.Equal, i - 1)
        i += 1

    for i in range(start_idx, len(out)):
        if (
            (
                (ca.real_body_gap_up(i - 1, i - 2) and ca.real_body_gap_up(i, i - 2))
                or (
                    ca.real_body_gap_down(i - 1, i - 2)
                    and ca.real_body_gap_down(i, i - 2)
                )
            )
            and ca.color[i - 1] == 1  # 2nd: white
            and ca.color[i] == 1  # 3rd: white
            and ca.real_body[i]
            >= ca.real_body[i - 1]
            - ca.candle_average(CandleSetting.Near, near_total, i - 1)  # same size
            and ca.real_body[i]
            <= ca.real_body[i - 1]
            + ca.candle_average(CandleSetting.Near, near_total, i - 1)
            and ca.open[i]
            >= ca.open[i - 1]
            - ca.candle_average(CandleSetting.Equal, equal_total, i - 1)  # same open
            and ca.open[i]
            <= ca.open[i - 1]
            + ca.candle_average(CandleSetting.Equal, equal_total, i - 1)
        ):
            out[i] = 100 if ca.real_body_gap_up(i - 1, i - 2) else -100

        near_total += ca.candle_range(CandleSetting.Near, i - 1) - ca.candle_range(
            CandleSetting.Near, near_trail - 1
        )
        equal_total += ca.candle_range(CandleSetting.Equal, i - 1) - ca.candle_range(
            CandleSetting.Equal, equal_trail - 1
        )
        near_trail += 1
        equal_trail += 1


def cdl_gapsidesidewhite(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Gap Side-by-Side White Lines"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_GAPSIDESIDEWHITE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
