# Candle Long Legged Doji (CDL_LONGLEGGEDDOJI)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    CandleSetting,
    candle_average,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(real_body, lower_shadow, upper_shadow, body_doji, shadow_long, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i] <= body_doji[i] and (lower_shadow[i] > shadow_long[i] or upper_shadow[i] > shadow_long[i]):
            out[i] = 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(CandleSetting.BodyDoji), candle_avg_period(CandleSetting.ShadowLong))
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 0, start_idx),
        out,
        start_idx,
    )


def cdl_longleggeddoji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
