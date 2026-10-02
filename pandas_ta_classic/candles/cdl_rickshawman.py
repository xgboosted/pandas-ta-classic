# Candle Rickshaw Man (CDL_RICKSHAWMAN)
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
def _detect_nb(real_body, lower_shadow, upper_shadow, body_hi, body_lo, low, hl_range, body_doji, shadow_long, near, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            real_body[i] <= body_doji[i]
            and lower_shadow[i] > shadow_long[i]
            and upper_shadow[i] > shadow_long[i]
            and (body_lo[i] <= low[i] + hl_range[i] / 2.0 + near[i])
            and (body_hi[i] >= low[i] + hl_range[i] / 2.0 - near[i])
        ):
            out[i] = 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.BodyDoji, CandleSetting.ShadowLong, CandleSetting.Near))
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        ca.body_high,
        ca.body_low,
        ca.low,
        ca.hl_range,
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 0, start_idx),
        candle_average(ca, CandleSetting.Near, 0, start_idx),
        out,
        start_idx,
    )


def cdl_rickshawman(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Rickshawman"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_RICKSHAWMAN",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
