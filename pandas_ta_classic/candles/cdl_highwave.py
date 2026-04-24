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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_vl_period = candle_avg_period(CandleSetting.ShadowVeryLong)
    lookback = max(body_short_period, shadow_vl_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_svl = ca._ranges[CandleSetting.ShadowVeryLong]

    body_short_trail = start_idx - body_short_period
    shadow_vl_trail = start_idx - shadow_vl_period
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    shadow_vl_total = float(arr_svl[shadow_vl_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and ca.upper_shadow[i]
            > AVG_FACTOR[CandleSetting.ShadowVeryLong] * arr_svl[i]
            and ca.lower_shadow[i]
            > AVG_FACTOR[CandleSetting.ShadowVeryLong] * arr_svl[i]
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        shadow_vl_total += arr_svl[i] - arr_svl[shadow_vl_trail]
        body_short_trail += 1
        shadow_vl_trail += 1


def cdl_highwave(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Highwave"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HIGHWAVE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
