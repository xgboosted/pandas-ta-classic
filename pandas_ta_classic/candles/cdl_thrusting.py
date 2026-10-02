# Candle Thrusting Pattern (CDL_THRUSTING)
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
def _detect_nb(color, real_body, open_, low, close, body_long_1, equal_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -1
            and real_body[i - 1] > body_long_1[i]
            and color[i] == 1
            and open_[i] < low[i - 1]
            and close[i] > close[i - 1] + equal_1[i]
            and close[i] <= open_[i - 1] - real_body[i - 1] * 0.5
        ):
            out[i] = -100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.Equal), candle_avg_period(CandleSetting.BodyLong)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.low,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.Equal, 1, start_idx),
        out,
        start_idx,
    )


def cdl_thrusting(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Thrusting"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_THRUSTING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
