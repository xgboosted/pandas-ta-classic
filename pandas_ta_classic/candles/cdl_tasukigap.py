# Candle Tasuki Gap (CDL_TASUKIGAP)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(
    color,
    real_body,
    open_,
    close,
    body_hi,
    body_lo,
    arr_nr,
    out,
    start_idx,
    near_trail,
    near_total,
    f_near,
):
    for i in range(start_idx, len(out)):
        if (
            body_lo[i - 1] > body_hi[i - 2]
            and color[i - 1] == 1
            and color[i] == -1
            and open_[i] < close[i - 1]
            and open_[i] > open_[i - 1]
            and close[i] < open_[i - 1]
            and close[i] > body_hi[i - 2]
            and abs(real_body[i - 1] - real_body[i]) < f_near * near_total
        ) or (
            body_hi[i - 1] < body_lo[i - 2]
            and color[i - 1] == -1
            and color[i] == 1
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]
            and close[i] > open_[i - 1]
            and close[i] < body_lo[i - 2]
            and abs(real_body[i - 1] - real_body[i]) < f_near * near_total
        ):
            out[i] = color[i - 1] * 100

        near_total += arr_nr[i - 1] - arr_nr[near_trail - 1]
        near_trail += 1


def _detect(ca, out, **kwargs):
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]
    body_hi = ca.body_high
    body_lo = ca.body_low

    near_trail = start_idx - near_period
    near_total = float(arr_nr[near_trail - 1 : start_idx - 1].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.close,
        body_hi,
        body_lo,
        arr_nr,
        out,
        start_idx,
        near_trail,
        near_total,
        AVG_FACTOR[CandleSetting.Near],
    )


def cdl_tasukigap(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Tasuki Gap"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TASUKIGAP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
