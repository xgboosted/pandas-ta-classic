# Candle Unique 3 River (CDL_UNIQUE3RIVER)
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
def _detect_nb(real_body, color, open_, low, close, body_long_2, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            real_body[i - 2] > body_long_2[i]
            and color[i - 2] == -1
            and color[i - 1] == -1
            and close[i - 1] > close[i - 2]
            and open_[i - 1] <= open_[i - 2]
            and low[i - 1] < low[i - 2]
            and real_body[i] < body_short[i]
            and color[i] == 1
            and open_[i] > low[i - 1]
        ):
            out[i] = 100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.BodyLong), candle_avg_period(CandleSetting.BodyShort)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.open,
        ca.low,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_unique3river(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Unique Three River"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_UNIQUE3RIVER",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
