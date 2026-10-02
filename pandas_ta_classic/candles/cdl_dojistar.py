# Candle Doji Star (CDL_DOJISTAR)
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
def _detect_nb(real_body, color, body_hi, body_lo, body_long_1, body_doji, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            real_body[i - 1] > body_long_1[i]
            and real_body[i] <= body_doji[i]
            and ((color[i - 1] == 1 and body_lo[i] > body_hi[i - 1]) or (color[i - 1] == -1 and body_hi[i] < body_lo[i - 1]))
        ):
            out[i] = -color[i - 1] * 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(CandleSetting.BodyLong), candle_avg_period(CandleSetting.BodyDoji)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        out,
        start_idx,
    )


def cdl_dojistar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Dojistar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DOJISTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
