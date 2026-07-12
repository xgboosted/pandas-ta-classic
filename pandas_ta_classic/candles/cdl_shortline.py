# Candle Short Line Candle (CDL_SHORTLINE)
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
    real_body,
    upper_shadow,
    lower_shadow,
    color,
    arr_bs,
    arr_ss,
    out,
    start_idx,
    body_short_trail,
    shadow_short_trail,
    body_short_total,
    shadow_short_total,
    f_bs,
    f_ss,
):
    for i in range(start_idx, len(out)):
        if real_body[i] < f_bs * body_short_total and upper_shadow[i] < f_ss * shadow_short_total and lower_shadow[i] < f_ss * shadow_short_total:
            out[i] = color[i] * 100

        # Update trailing windows
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        shadow_short_total += arr_ss[i] - arr_ss[shadow_short_trail]
        body_short_trail += 1
        shadow_short_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_short_period = candle_avg_period(CandleSetting.ShadowShort)
    lookback = max(body_short_period, shadow_short_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_ss = ca._ranges[CandleSetting.ShadowShort]

    body_short_trail = start_idx - body_short_period
    shadow_short_trail = start_idx - shadow_short_period
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    shadow_short_total = float(arr_ss[shadow_short_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        ca.color,
        arr_bs,
        arr_ss,
        out,
        start_idx,
        body_short_trail,
        shadow_short_trail,
        body_short_total,
        shadow_short_total,
        AVG_FACTOR[CandleSetting.BodyShort],
        AVG_FACTOR[CandleSetting.ShadowShort],
    )


def cdl_shortline(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Shortline"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SHORTLINE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
