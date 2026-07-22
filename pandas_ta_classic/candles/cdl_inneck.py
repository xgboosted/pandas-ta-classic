# Candle In-Neck Pattern (CDL_INNECK)
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
    low,
    close,
    arr_bl,
    arr_eq,
    out,
    start_idx,
    equal_trail,
    body_long_trail,
    equal_total,
    body_long_total,
    f_bl,
    f_eq,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -1
            and real_body[i - 1] > f_bl * body_long_total
            and color[i] == 1
            and open_[i] < low[i - 1]
            and close[i] <= close[i - 1] + f_eq * equal_total
            and close[i] >= close[i - 1]
        ):
            out[i] = -100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        equal_trail += 1
        body_long_trail += 1


def _detect(ca, out, **kwargs):
    equal_period = candle_avg_period(CandleSetting.Equal)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 1 - equal_period
    body_long_trail = start_idx - 1 - body_long_period
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.low,
        ca.close,
        arr_bl,
        arr_eq,
        out,
        start_idx,
        equal_trail,
        body_long_trail,
        equal_total,
        body_long_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.Equal],
    )


def cdl_inneck(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Inneck"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_INNECK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
