# Candle Counterattack (CDL_COUNTERATTACK)
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
def _detect_nb(color, real_body, close, body_long_1, body_long_0, equal_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]  # opposite candles
            and real_body[i - 1] > body_long_1[i]  # 1st long
            and real_body[i] > body_long_0[i]  # 2nd long
            and close[i] <= close[i - 1] + equal_1[i]  # equal closes
            and close[i] >= close[i - 1] - equal_1[i]
        ):
            out[i] = color[i] * 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(Equal, BodyLong) + 1
    start_idx = max(candle_avg_period(CandleSetting.Equal), candle_avg_period(CandleSetting.BodyLong)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyLong, 0, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.Equal, 1, start_idx, sequential_seed=True),
        out,
        start_idx,
    )


def cdl_counterattack(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Counterattack"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_COUNTERATTACK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
