# Candle Closing Marubozu (CDL_CLOSINGMARUBOZU)
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
def _detect_nb(real_body, upper_shadow, lower_shadow, color, body_long, shadow_very_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i] > body_long[i] and (
            (color[i] == 1 and upper_shadow[i] < shadow_very_short[i]) or (color[i] == -1 and lower_shadow[i] < shadow_very_short[i])
        ):
            out[i] = color[i] * 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(CandleSetting.BodyLong), candle_avg_period(CandleSetting.ShadowVeryShort))
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        ca.color,
        candle_average(ca, CandleSetting.BodyLong, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_closingmarubozu(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Closingmarubozu"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_CLOSINGMARUBOZU",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
