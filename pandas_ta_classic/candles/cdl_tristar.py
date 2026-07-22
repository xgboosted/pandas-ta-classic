# Candle Tristar Pattern (CDL_TRISTAR)
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
    real_body,
    open_,
    close,
    body_hi,
    body_lo,
    arr_bd,
    out,
    start_idx,
    body_trail,
    body_total,
    f_bd,
):
    for i in range(start_idx, len(out)):
        if real_body[i - 2] <= f_bd * body_total and real_body[i - 1] <= f_bd * body_total and real_body[i] <= f_bd * body_total:
            if body_lo[i - 1] > body_hi[i - 2] and body_hi[i] < max(open_[i - 1], close[i - 1]):
                out[i] = -100
            if body_hi[i - 1] < body_lo[i - 2] and min(open_[i], close[i]) > body_lo[i - 1]:
                out[i] = 100

        body_total += arr_bd[i - 2] - arr_bd[body_trail]
        body_trail += 1


def _detect(ca, out, **kwargs):
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = body_doji_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_trail = start_idx - 2 - body_doji_period
    body_total = float(arr_bd[body_trail : start_idx - 2].sum())

    _detect_nb(
        ca.real_body,
        ca.open,
        ca.close,
        body_hi,
        body_lo,
        arr_bd,
        out,
        start_idx,
        body_trail,
        body_total,
        AVG_FACTOR[CandleSetting.BodyDoji],
    )


def cdl_tristar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Tristar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TRISTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
