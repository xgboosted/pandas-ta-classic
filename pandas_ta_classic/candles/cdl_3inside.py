# Candle Three Inside Up/Down (CDL_3INSIDE)
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
    close,
    open_,
    body_hi,
    body_lo,
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
            and real_body[i - 1] <= f_bs * body_short_total
            and body_hi[i - 1] < body_hi[i - 2]
            and body_lo[i - 1] > body_lo[i - 2]
            and (
                (color[i - 2] == 1 and color[i] == -1 and close[i] < open_[i - 2])
                or (color[i - 2] == -1 and color[i] == 1 and close[i] > open_[i - 2])
            )
        ):
            out[i] = -color[i - 2] * 100

        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i - 1] - arr_bs[body_short_trail]
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
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 2 - body_long_period
    body_short_trail = start_idx - 1 - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_short_total = float(arr_bs[body_short_trail : start_idx - 1].sum())

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.close,
        ca.open,
        body_hi,
        body_lo,
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


def cdl_3inside(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Three Inside Up/Down"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3INSIDE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
