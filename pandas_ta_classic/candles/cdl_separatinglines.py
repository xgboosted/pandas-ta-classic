# Candle Separating Lines (CDL_SEPARATINGLINES)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(
    color,
    real_body,
    open_,
    upper_shadow,
    lower_shadow,
    arr_bl,
    arr_eq,
    arr_svs,
    out,
    start_idx,
    shadow_vs_trail,
    body_long_trail,
    equal_trail,
    shadow_vs_total,
    body_long_total,
    equal_total,
    f_bl,
    f_eq,
    f_svs,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]
            and open_[i] <= open_[i - 1] + f_eq * equal_total
            and open_[i] >= open_[i - 1] - f_eq * equal_total
            and real_body[i] > f_bl * body_long_total
            and ((color[i] == 1 and lower_shadow[i] < f_svs * shadow_vs_total) or (color[i] == -1 and upper_shadow[i] < f_svs * shadow_vs_total))
        ):
            out[i] = color[i] * 100

        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        body_long_total += arr_bl[i] - arr_bl[body_long_trail]
        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        shadow_vs_trail += 1
        body_long_trail += 1
        equal_trail += 1


def _detect(ca, out, **kwargs):
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = max(shadow_vs_period, body_long_period, equal_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_trail = start_idx - body_long_period
    equal_trail = start_idx - 1 - equal_period
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    body_long_total = float(arr_bl[body_long_trail:start_idx].sum())
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.upper_shadow,
        ca.lower_shadow,
        arr_bl,
        arr_eq,
        arr_svs,
        out,
        start_idx,
        shadow_vs_trail,
        body_long_trail,
        equal_trail,
        shadow_vs_total,
        body_long_total,
        equal_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.Equal],
        AVG_FACTOR[CandleSetting.ShadowVeryShort],
    )


def cdl_separatinglines(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
