# Candle Dragonfly Doji (CDL_DRAGONFLYDOJI)
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
    upper_shadow,
    lower_shadow,
    arr_bd,
    arr_svs,
    out,
    start_idx,
    body_doji_trail,
    shadow_vs_trail,
    body_doji_total,
    shadow_vs_total,
    f_bd,
    f_svs,
):
    for i in range(start_idx, len(out)):
        if real_body[i] <= f_bd * body_doji_total and upper_shadow[i] < f_svs * shadow_vs_total and lower_shadow[i] > f_svs * shadow_vs_total:
            out[i] = 100

        # Update trailing windows
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        body_doji_trail += 1
        shadow_vs_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = max(body_doji_period, shadow_vs_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    body_doji_trail = start_idx - body_doji_period
    shadow_vs_trail = start_idx - shadow_vs_period
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        arr_bd,
        arr_svs,
        out,
        start_idx,
        body_doji_trail,
        shadow_vs_trail,
        body_doji_total,
        shadow_vs_total,
        AVG_FACTOR[CandleSetting.BodyDoji],
        AVG_FACTOR[CandleSetting.ShadowVeryShort],
    )


def cdl_dragonflydoji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Dragonflydoji"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DRAGONFLYDOJI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
