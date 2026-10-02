# Candle Up/Down-gap side-by-side white lines (CDL_GAPSIDESIDEWHITE)
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
def _detect_nb(color, real_body, open_, body_hi, body_lo, near_1, equal_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            ((body_lo[i - 1] > body_hi[i - 2] and body_lo[i] > body_hi[i - 2]) or (body_hi[i - 1] < body_lo[i - 2] and body_hi[i] < body_lo[i - 2]))
            and color[i - 1] == 1  # 2nd: white
            and color[i] == 1  # 3rd: white
            and real_body[i] >= real_body[i - 1] - near_1[i]  # same size
            and real_body[i] <= real_body[i - 1] + near_1[i]
            and open_[i] >= open_[i - 1] - equal_1[i]  # same open
            and open_[i] <= open_[i - 1] + equal_1[i]
        ):
            out[i] = 100 if body_lo[i - 1] > body_hi[i - 2] else -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(Near, Equal) + 2
    start_idx = max(candle_avg_period(CandleSetting.Near), candle_avg_period(CandleSetting.Equal)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.Near, 1, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.Equal, 1, start_idx, sequential_seed=True),
        out,
        start_idx,
    )


def cdl_gapsidesidewhite(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Gap Side-by-Side White Lines"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_GAPSIDESIDEWHITE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
