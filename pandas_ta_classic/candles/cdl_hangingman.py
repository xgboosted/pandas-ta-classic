# Candle Hanging Man (CDL_HANGINGMAN)
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
def _detect_nb(real_body, lower_shadow, upper_shadow, body_lo, high, body_short, shadow_long, shadow_very_short, near_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            real_body[i] < body_short[i]
            and lower_shadow[i] > shadow_long[i]
            and upper_shadow[i] < shadow_very_short[i]
            and body_lo[i] >= high[i - 1] - near_1[i]
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    settings = (CandleSetting.BodyShort, CandleSetting.ShadowLong, CandleSetting.ShadowVeryShort, CandleSetting.Near)
    start_idx = max(candle_avg_period(s) for s in settings) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        ca.body_low,
        ca.high,
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        candle_average(ca, CandleSetting.Near, 1, start_idx),
        out,
        start_idx,
    )


def cdl_hangingman(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Hangingman"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HANGINGMAN",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
