# Candle Long Legged Doji (CDL_LONGLEGGEDDOJI)
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
    lower_shadow,
    upper_shadow,
    arr_bd,
    arr_sl,
    out,
    start_idx,
    body_doji_trail,
    shadow_long_trail,
    body_doji_total,
    shadow_long_total,
    f_bd,
    f_sl,
):
    for i in range(start_idx, len(out)):
        if real_body[i] <= f_bd * body_doji_total and (lower_shadow[i] > f_sl * arr_sl[i] or upper_shadow[i] > f_sl * arr_sl[i]):
            out[i] = 100

        # Update trailing windows
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        body_doji_trail += 1
        shadow_long_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    lookback = max(body_doji_period, shadow_long_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]

    body_doji_trail = start_idx - body_doji_period
    shadow_long_trail = start_idx - shadow_long_period
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        arr_bd,
        arr_sl,
        out,
        start_idx,
        body_doji_trail,
        shadow_long_trail,
        body_doji_total,
        shadow_long_total,
        AVG_FACTOR[CandleSetting.BodyDoji],
        AVG_FACTOR[CandleSetting.ShadowLong],
    )


def cdl_longleggeddoji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Longleggeddoji"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_LONGLEGGEDDOJI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
