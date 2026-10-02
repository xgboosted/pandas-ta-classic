# Candle Upside Gap Two Crows (CDL_UPSIDEGAP2CROWS)
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
def _detect_nb(color, real_body, open_, close, body_hi, body_lo, body_long_2, body_short_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 2] == 1
            and real_body[i - 2] > body_long_2[i]
            and color[i - 1] == -1
            and real_body[i - 1] <= body_short_1[i]
            and body_lo[i - 1] > body_hi[i - 2]
            and color[i] == -1
            and open_[i] > open_[i - 1]
            and close[i] < close[i - 1]
            and close[i] > close[i - 2]
        ):
            out[i] = -100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.BodyLong), candle_avg_period(CandleSetting.BodyShort)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 1, start_idx),
        out,
        start_idx,
    )


def cdl_upsidegap2crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Upside Gap Two Crows"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_UPSIDEGAP2CROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
