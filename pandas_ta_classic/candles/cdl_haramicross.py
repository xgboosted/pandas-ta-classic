# Candle Harami Cross Pattern (CDL_HARAMICROSS)
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
def _detect_nb(real_body, color, body_hi, body_lo, close, open_, body_long_1, body_doji, out, start_idx):
    for i in range(start_idx, len(out)):
        if real_body[i - 1] > body_long_1[i] and real_body[i] <= body_doji[i]:
            if body_hi[i] < max(close[i - 1], open_[i - 1]) and body_lo[i] > body_lo[i - 1]:
                out[i] = -color[i - 1] * 100
            elif body_hi[i] <= max(close[i - 1], open_[i - 1]) and body_lo[i] >= body_lo[i - 1]:
                out[i] = -color[i - 1] * 80


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.BodyLong), candle_avg_period(CandleSetting.BodyDoji)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.body_high,
        ca.body_low,
        ca.close,
        ca.open,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyDoji, 0, start_idx),
        out,
        start_idx,
    )


def cdl_haramicross(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Haramicross"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HARAMICROSS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
