# Candle Counterattack (CDL_COUNTERATTACK)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit
import numpy as np


@njit(cache=True)
def _detect_nb(
    color,
    real_body,
    close,
    arr_bl,
    arr_eq,
    out,
    start_idx,
    equal_trail,
    body_long_trail,
    equal_total,
    body_long_total_1,
    body_long_total_0,
    f_bl,
    f_eq,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]  # opposite candles
            and real_body[i - 1] > f_bl * body_long_total_1  # 1st long
            and real_body[i] > f_bl * body_long_total_0  # 2nd long
            and close[i] <= close[i - 1] + f_eq * equal_total  # equal closes
            and close[i] >= close[i - 1] - f_eq * equal_total
        ):
            out[i] = color[i] * 100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail - 1]
        body_long_total_1 += arr_bl[i - 1] - arr_bl[body_long_trail - 1]
        body_long_total_0 += arr_bl[i] - arr_bl[body_long_trail]
        equal_trail += 1
        body_long_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(Equal, BodyLong) + 1
    equal_period = candle_avg_period(CandleSetting.Equal)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - equal_period
    body_long_trail = start_idx - body_long_period

    equal_total = 0.0
    i = equal_trail
    while i < start_idx:
        equal_total += arr_eq[i - 1]
        i += 1

    body_long_total_1 = 0.0
    body_long_total_0 = 0.0
    i = body_long_trail
    while i < start_idx:
        body_long_total_1 += arr_bl[i - 1]
        body_long_total_0 += arr_bl[i]
        i += 1

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.close,
        arr_bl,
        arr_eq,
        out,
        start_idx,
        equal_trail,
        body_long_trail,
        equal_total,
        body_long_total_1,
        body_long_total_0,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.Equal],
    )


def cdl_counterattack(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Counterattack"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_COUNTERATTACK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
