# Candle Homing Pigeon (CDL_HOMINGPIGEON)
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
def _detect_nb(color, real_body, open_, close, body_long_1, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -1
            and color[i] == -1
            and real_body[i - 1] > body_long_1[i]
            and real_body[i] <= body_short[i]
            and open_[i] < open_[i - 1]
            and close[i] > close[i - 1]
        ):
            out[i] = 100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.BodyShort), candle_avg_period(CandleSetting.BodyLong)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_homingpigeon(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Homingpigeon"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HOMINGPIGEON",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
