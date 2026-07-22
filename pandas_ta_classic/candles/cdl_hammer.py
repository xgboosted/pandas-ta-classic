# Candle Hammer (CDL_HAMMER)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit
import numpy as np


@njit(cache=True)
def _detect_nb(
    real_body,
    lower_shadow,
    upper_shadow,
    body_lo,
    low,
    arr_bs,
    arr_sl,
    arr_svs,
    arr_nr,
    out,
    start_idx,
    body_trail,
    shadow_long_trail,
    shadow_vs_trail,
    near_trail,
    body_total,
    shadow_long_total,
    shadow_vs_total,
    near_total,
    f_bs,
    f_sl,
    f_svs,
    f_near,
):
    for i in range(start_idx, len(out)):
        if (
            real_body[i] < f_bs * body_total
            and lower_shadow[i] > f_sl * arr_sl[i]
            and upper_shadow[i] < f_svs * shadow_vs_total
            and (body_lo[i] <= low[i - 1] + f_near * near_total)
        ):
            out[i] = 100

        # Update trailing windows AFTER pattern check
        body_total += arr_bs[i] - arr_bs[body_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        near_total += arr_nr[i - 1] - arr_nr[near_trail]

        body_trail += 1
        shadow_long_trail += 1
        shadow_vs_trail += 1
        near_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    body_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)  # 0
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    near_period = candle_avg_period(CandleSetting.Near)

    # Lookback: max of all periods + 1 (extra bar for i-1 reference)
    lookback = max(body_period, shadow_long_period, shadow_vs_period, near_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_nr = ca._ranges[CandleSetting.Near]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    body_lo = ca.body_low

    # Trailing indices
    body_trail = start_idx - body_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    near_trail = start_idx - 1 - near_period  # Near uses i-1

    # Seed totals
    body_total = float(arr_bs[body_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    near_total = float(arr_nr[near_trail : start_idx - 1].sum())

    _detect_nb(
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        body_lo,
        ca.low,
        arr_bs,
        arr_sl,
        arr_svs,
        arr_nr,
        out,
        start_idx,
        body_trail,
        shadow_long_trail,
        shadow_vs_trail,
        near_trail,
        body_total,
        shadow_long_total,
        shadow_vs_total,
        near_total,
        AVG_FACTOR[CandleSetting.BodyShort],
        AVG_FACTOR[CandleSetting.ShadowLong],
        AVG_FACTOR[CandleSetting.ShadowVeryShort],
        AVG_FACTOR[CandleSetting.Near],
    )


def cdl_hammer(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Hammer"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HAMMER",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
