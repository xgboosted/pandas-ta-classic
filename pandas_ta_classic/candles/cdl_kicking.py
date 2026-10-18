# Candle Kicking (CDL_KICKING)
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
def _detect_nb(color, real_body, upper_shadow, lower_shadow, hi, lo, body_long_1, body_long_0, svs_1, svs_0, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]
            and real_body[i - 1] > body_long_1[i]
            and upper_shadow[i - 1] < svs_1[i]
            and lower_shadow[i - 1] < svs_1[i]
            and real_body[i] > body_long_0[i]
            and upper_shadow[i] < svs_0[i]
            and lower_shadow[i] < svs_0[i]
            and ((color[i - 1] == -1 and lo[i] > hi[i - 1]) or (color[i - 1] == 1 and hi[i] < lo[i - 1]))
        ):
            out[i] = color[i] * 100


def _detect(ca, out, **kwargs):
    start_idx = max(candle_avg_period(CandleSetting.ShadowVeryShort), candle_avg_period(CandleSetting.BodyLong)) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        ca.high,
        ca.low,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyLong, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_kicking(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Kicking"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_KICKING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
