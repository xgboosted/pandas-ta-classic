# Candle Tasuki Gap (CDL_TASUKIGAP)
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
def _detect_nb(color, real_body, open_, close, body_hi, body_lo, near_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            body_lo[i - 1] > body_hi[i - 2]
            and color[i - 1] == 1
            and color[i] == -1
            and open_[i] < close[i - 1]
            and open_[i] > open_[i - 1]
            and close[i] < open_[i - 1]
            and close[i] > body_hi[i - 2]
            and abs(real_body[i - 1] - real_body[i]) < near_1[i]
        ) or (
            body_hi[i - 1] < body_lo[i - 2]
            and color[i - 1] == -1
            and color[i] == 1
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]
            and close[i] > open_[i - 1]
            and close[i] < body_lo[i - 2]
            and abs(real_body[i - 1] - real_body[i]) < near_1[i]
        ):
            out[i] = color[i - 1] * 100


def _detect(ca, out, **kwargs):
    start_idx = candle_avg_period(CandleSetting.Near) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.Near, 1, start_idx),
        out,
        start_idx,
    )


def cdl_tasukigap(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Tasuki Gap"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TASUKIGAP",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
