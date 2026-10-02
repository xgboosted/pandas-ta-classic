# Candle Tristar Pattern (CDL_TRISTAR)
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
def _detect_nb(real_body, open_, close, body_hi, body_lo, body_doji_2, out, start_idx):
    for i in range(start_idx, len(out)):
        # TA-Lib checks all three candles against the average before the 1st
        if real_body[i - 2] <= body_doji_2[i] and real_body[i - 1] <= body_doji_2[i] and real_body[i] <= body_doji_2[i]:
            if body_lo[i - 1] > body_hi[i - 2] and body_hi[i] < max(open_[i - 1], close[i - 1]):
                out[i] = -100
            if body_hi[i - 1] < body_lo[i - 2] and min(open_[i], close[i]) > body_lo[i - 1]:
                out[i] = 100


def _detect(ca, out, **kwargs):
    start_idx = candle_avg_period(CandleSetting.BodyDoji) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.open,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyDoji, 2, start_idx),
        out,
        start_idx,
    )


def cdl_tristar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Tristar"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TRISTAR",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
