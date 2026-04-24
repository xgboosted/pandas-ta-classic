# -*- coding: utf-8 -*-
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
import numpy as np


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    shadow_vl_period = candle_avg_period(CandleSetting.ShadowVeryLong)
    lookback = max(body_doji_period, shadow_vs_period, shadow_vl_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_svl = ca._ranges[CandleSetting.ShadowVeryLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    body_doji_trail = start_idx - body_doji_period
    shadow_vs_trail = start_idx - shadow_vs_period
    shadow_vl_trail = start_idx - shadow_vl_period
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    shadow_vl_total = float(arr_svl[shadow_vl_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_doji_total
            and ca.upper_shadow[i]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            and ca.lower_shadow[i]
            > AVG_FACTOR[CandleSetting.ShadowVeryLong] * arr_svl[i]
        ):
            out[i] = 100

        # Update trailing windows
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        shadow_vl_total += arr_svl[i] - arr_svl[shadow_vl_trail]
        body_doji_trail += 1
        shadow_vs_trail += 1
        shadow_vl_trail += 1


def cdl_takuri(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Takuri"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TAKURI",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
