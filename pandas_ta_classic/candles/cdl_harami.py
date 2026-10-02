# Candle Harami Pattern (CDL_HARAMI)
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
def _detect_nb(real_body, color, body_hi, body_lo, body_long_1, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i - 1] > body_long_1[i] and real_body[i] <= body_short[i]:  # 1st: long  # 2nd: short
            hi_i = body_hi[i]
            lo_i = body_lo[i]
            hi_p = body_hi[i - 1]
            lo_p = body_lo[i - 1]
            if hi_i < hi_p and lo_i > lo_p:
                out[i] = -color[i - 1] * 100
            elif hi_i <= hi_p and lo_i >= lo_p:
                out[i] = -color[i - 1] * 80


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(BodyShort, BodyLong) + 1
    start_idx = max(candle_avg_period(CandleSetting.BodyShort), candle_avg_period(CandleSetting.BodyLong)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_harami(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Harami"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HARAMI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
