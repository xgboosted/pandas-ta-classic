# Candle Up/Down-gap side-by-side white lines (CDL_GAPSIDESIDEWHITE)
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
    open_,
    body_hi,
    body_lo,
    arr_eq,
    arr_nr,
    out,
    start_idx,
    near_trail,
    equal_trail,
    near_total,
    equal_total,
    f_near,
    f_eq,
):
    for i in range(start_idx, len(out)):
        if (
            ((body_lo[i - 1] > body_hi[i - 2] and body_lo[i] > body_hi[i - 2]) or (body_hi[i - 1] < body_lo[i - 2] and body_hi[i] < body_lo[i - 2]))
            and color[i - 1] == 1  # 2nd: white
            and color[i] == 1  # 3rd: white
            and real_body[i] >= real_body[i - 1] - f_near * near_total  # same size
            and real_body[i] <= real_body[i - 1] + f_near * near_total
            and open_[i] >= open_[i - 1] - f_eq * equal_total  # same open
            and open_[i] <= open_[i - 1] + f_eq * equal_total
        ):
            out[i] = 100 if body_lo[i - 1] > body_hi[i - 2] else -100

        near_total += arr_nr[i - 1] - arr_nr[near_trail - 1]
        equal_total += arr_eq[i - 1] - arr_eq[equal_trail - 1]
        near_trail += 1
        equal_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(Near, Equal) + 2
    near_period = candle_avg_period(CandleSetting.Near)
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(near_period, equal_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_eq = ca._ranges[CandleSetting.Equal]
    arr_nr = ca._ranges[CandleSetting.Near]
    body_hi = ca.body_high
    body_lo = ca.body_low

    near_trail = start_idx - near_period
    equal_trail = start_idx - equal_period

    near_total = 0.0
    i = near_trail
    while i < start_idx:
        near_total += arr_nr[i - 1]
        i += 1

    equal_total = 0.0
    i = equal_trail
    while i < start_idx:
        equal_total += arr_eq[i - 1]
        i += 1

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        body_hi,
        body_lo,
        arr_eq,
        arr_nr,
        out,
        start_idx,
        near_trail,
        equal_trail,
        near_total,
        equal_total,
        AVG_FACTOR[CandleSetting.Near],
        AVG_FACTOR[CandleSetting.Equal],
    )


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
