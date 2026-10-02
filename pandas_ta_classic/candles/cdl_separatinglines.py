# Candle Separating Lines (CDL_SEPARATINGLINES)
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
def _detect_nb(color, real_body, open_, upper_shadow, lower_shadow, body_long, equal_1, shadow_very_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]
            and open_[i] <= open_[i - 1] + equal_1[i]
            and open_[i] >= open_[i - 1] - equal_1[i]
            and real_body[i] > body_long[i]
            and ((color[i] == 1 and lower_shadow[i] < shadow_very_short[i]) or (color[i] == -1 and upper_shadow[i] < shadow_very_short[i]))
        ):
            out[i] = color[i] * 100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.ShadowVeryShort, CandleSetting.BodyLong, CandleSetting.Equal)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.upper_shadow,
        ca.lower_shadow,
        candle_average(ca, CandleSetting.BodyLong, 0, start_idx),
        candle_average(ca, CandleSetting.Equal, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_separatinglines(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Separatinglines"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SEPARATINGLINES",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
