# Candle Unique 3 River (CDL_UNIQUE3RIVER)
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
    open_,
    low,
    close,
    arr_bl,
    arr_bs,
    out,
    start_idx,
    body_long_trail,
    body_short_trail,
    body_long_total,
    body_short_total,
    f_bl,
    f_bs,
):
    for i in range(start_idx, len(out)):
        if (
            real_body[i - 2] > f_bl * body_long_total
            and color[i - 2] == -1
            and color[i - 1] == -1
            and close[i - 1] > close[i - 2]
            and open_[i - 1] <= open_[i - 2]
            and low[i - 1] < low[i - 2]
            and real_body[i] < f_bs * body_short_total
            and color[i] == 1
            and open_[i] > low[i - 1]
        ):
            out[i] = 100

        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_short_trail += 1


def _detect(ca, out, **kwargs):
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]

    body_long_trail = start_idx - 2 - body_long_period
    body_short_trail = start_idx - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.open,
        ca.low,
        ca.close,
        arr_bl,
        arr_bs,
        out,
        start_idx,
        body_long_trail,
        body_short_trail,
        body_long_total,
        body_short_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.BodyShort],
    )


def cdl_unique3river(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Unique Three River"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_UNIQUE3RIVER",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
