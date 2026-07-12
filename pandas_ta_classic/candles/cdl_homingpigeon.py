# Candle Homing Pigeon (CDL_HOMINGPIGEON)
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
            color[i - 1] == -1
            and color[i] == -1
            and real_body[i - 1] > f_bl * body_long_total
            and real_body[i] <= f_bs * body_short_total
            and open_[i] < open_[i - 1]
            and close[i] > close[i - 1]
        ):
            out[i] = 100

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_short_trail += 1


def _detect(ca, out, **kwargs):
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]

    body_long_trail = start_idx - 1 - body_long_period
    body_short_trail = start_idx - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
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


def cdl_homingpigeon(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Homingpigeon"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HOMINGPIGEON",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
