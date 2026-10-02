# Candle Takuri (Dragonfly Doji with very long lower shadow) (CDL_TAKURI)
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
def _detect_nb(real_body, upper_shadow, lower_shadow, body_doji, shadow_very_short, shadow_very_long, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i] <= body_doji[i] and upper_shadow[i] < shadow_very_short[i] and lower_shadow[i] > shadow_very_long[i]:
            out[i] = 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.BodyDoji, CandleSetting.ShadowVeryShort, CandleSetting.ShadowVeryLong))
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryLong, 0, start_idx),
        out,
        start_idx,
    )


def cdl_takuri(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Takuri"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TAKURI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
