# Candle Spinning Top (CDL_SPINNINGTOP)
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
def _detect_nb(real_body, upper_shadow, lower_shadow, color, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i] < body_short[i] and upper_shadow[i] > real_body[i] and lower_shadow[i] > real_body[i]:
            out[i] = color[i] * 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = candle_avg_period(CandleSetting.BodyShort)
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        ca.color,
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_spinningtop(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Spinningtop"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SPINNINGTOP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
