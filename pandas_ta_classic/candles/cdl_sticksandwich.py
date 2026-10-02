# Candle Stick Sandwich (CDL_STICKSANDWICH)
from typing import Any

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleSetting,
    candle_average,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(color, low, close, equal_2, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 2] == -1
            and color[i - 1] == 1
            and color[i] == -1
            and low[i - 1] > close[i - 2]
            and close[i] <= close[i - 2] + equal_2[i]
            and close[i] >= close[i - 2] - equal_2[i]
        ):
            out[i] = 100


def _detect(ca, out, **kwargs):
    start_idx = candle_avg_period(CandleSetting.Equal) + 2
    if start_idx >= len(out):
        return

    _detect_nb(ca.color, ca.low, ca.close, candle_average(ca, CandleSetting.Equal, 2, start_idx), out, start_idx)


def cdl_sticksandwich(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Sticksandwich"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_STICKSANDWICH",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
