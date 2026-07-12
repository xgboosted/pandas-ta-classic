# Candle Piercing Pattern (CDL_PIERCING)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
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
    low,
    close,
    arr_bl,
    body_long_total,
    out,
    start_idx,
    body_long_trail,
    f_bl,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -1
            and real_body[i - 1] > f_bl * body_long_total[1]
            and color[i] == 1
            and real_body[i] > f_bl * body_long_total[0]
            and open_[i] < low[i - 1]
            and close[i] < open_[i - 1]
            and close[i] > close[i - 1] + real_body[i - 1] * 0.5
        ):
            out[i] = 100

        for tot_idx in range(2):
            body_long_total[tot_idx] += arr_bl[i - tot_idx] - arr_bl[body_long_trail - tot_idx]
        body_long_trail += 1


def _detect(ca, out, **kwargs):
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]

    body_long_trail = start_idx - body_long_period

    body_long_total = np.array([0.0, 0.0])
    for j in range(body_long_trail, start_idx):
        body_long_total[1] += arr_bl[j - 1]
        body_long_total[0] += arr_bl[j]

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.low,
        ca.close,
        arr_bl,
        body_long_total,
        out,
        start_idx,
        body_long_trail,
        AVG_FACTOR[CandleSetting.BodyLong],
    )


def cdl_piercing(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Piercing"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_PIERCING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
