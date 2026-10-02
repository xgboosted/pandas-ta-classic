# Candle Shooting Star (CDL_SHOOTINGSTAR)
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
def _detect_nb(real_body, upper_shadow, lower_shadow, body_hi, body_lo, body_short, shadow_long, shadow_very_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            real_body[i] < body_short[i]
            and upper_shadow[i] > shadow_long[i]
            and lower_shadow[i] < shadow_very_short[i]
            and body_lo[i] > body_hi[i - 1]
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.BodyShort, CandleSetting.ShadowLong, CandleSetting.ShadowVeryShort)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_shootingstar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Shootingstar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SHOOTINGSTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
