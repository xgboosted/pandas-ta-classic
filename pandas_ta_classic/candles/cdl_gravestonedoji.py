# Candle Gravestone Doji (CDL_GRAVESTONEDOJI)
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
def _detect_nb(real_body, lower_shadow, upper_shadow, body_doji, shadow_very_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i] <= body_doji[i] and lower_shadow[i] < shadow_very_short[i] and upper_shadow[i] > shadow_very_short[i]:
            out[i] = 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(CandleSetting.BodyDoji), candle_avg_period(CandleSetting.ShadowVeryShort))
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_gravestonedoji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Gravestonedoji"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_GRAVESTONEDOJI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
