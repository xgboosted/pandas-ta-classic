# Candle Matching Low (CDL_MATCHINGLOW)
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
def _detect_nb(color, close, arr_eq, out, start_idx, equal_trail, equal_total, f_eq):
    for i in range(start_idx, len(out)):
        if color[i - 1] == -1 and color[i] == -1 and close[i] <= close[i - 1] + f_eq * equal_total and close[i] >= close[i - 1] - f_eq * equal_total:
            out[i] = 100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        equal_trail += 1


def _detect(ca, out, **kwargs):
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = equal_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 1 - equal_period
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())

    _detect_nb(
        ca.color,
        ca.close,
        arr_eq,
        out,
        start_idx,
        equal_trail,
        equal_total,
        AVG_FACTOR[CandleSetting.Equal],
    )


def cdl_matchinglow(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Matchinglow"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_MATCHINGLOW",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
