# Candle Harami Cross Pattern (CDL_HARAMICROSS)
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
    color,
    body_hi,
    body_lo,
    close,
    open_,
    arr_bl,
    arr_bd,
    out,
    start_idx,
    body_long_trail,
    body_doji_trail,
    body_long_total,
    body_doji_total,
    f_bl,
    f_bd,
):
    for i in range(start_idx, len(out)):
        if real_body[i - 1] > f_bl * body_long_total and real_body[i] <= f_bd * body_doji_total:
            if body_hi[i] < max(close[i - 1], open_[i - 1]) and body_lo[i] > body_lo[i - 1]:
                out[i] = -color[i - 1] * 100
            elif body_hi[i] <= max(close[i - 1], open_[i - 1]) and body_lo[i] >= body_lo[i - 1]:
                out[i] = -color[i - 1] * 80

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        body_long_trail += 1
        body_doji_trail += 1


def _detect(ca, out, **kwargs):
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    lookback = max(body_long_period, body_doji_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_bl = ca._ranges[CandleSetting.BodyLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 1 - body_long_period
    body_doji_trail = start_idx - body_doji_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.color,
        body_hi,
        body_lo,
        ca.close,
        ca.open,
        arr_bl,
        arr_bd,
        out,
        start_idx,
        body_long_trail,
        body_doji_trail,
        body_long_total,
        body_doji_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.BodyDoji],
    )


def cdl_haramicross(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Haramicross"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HARAMICROSS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
