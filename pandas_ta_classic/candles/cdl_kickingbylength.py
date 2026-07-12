# Candle Kicking - bull/bear determined by the longer marubozu (CDL_KICKINGBYLENGTH)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    AVG_FACTOR,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit
import numpy as np


@njit(cache=True)
def _detect_nb(
    color,
    real_body,
    upper_shadow,
    lower_shadow,
    hi,
    lo,
    arr_bl,
    arr_svs,
    shadow_vs_total,
    body_long_total,
    out,
    start_idx,
    shadow_vs_trail,
    body_long_trail,
    f_bl,
    f_svs,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == -color[i]
            and real_body[i - 1] > f_bl * body_long_total[1]
            and upper_shadow[i - 1] < f_svs * shadow_vs_total[1]
            and lower_shadow[i - 1] < f_svs * shadow_vs_total[1]
            and real_body[i] > f_bl * body_long_total[0]
            and upper_shadow[i] < f_svs * shadow_vs_total[0]
            and lower_shadow[i] < f_svs * shadow_vs_total[0]
            and ((color[i - 1] == -1 and lo[i] > hi[i - 1]) or (color[i - 1] == 1 and hi[i] < lo[i - 1]))
        ):
            out[i] = color[i if real_body[i] > real_body[i - 1] else i - 1] * 100

        for tot_idx in range(2):
            body_long_total[tot_idx] += arr_bl[i - tot_idx] - arr_bl[body_long_trail - tot_idx]
            shadow_vs_total[tot_idx] += arr_svs[i - tot_idx] - arr_svs[shadow_vs_trail - tot_idx]
        body_long_trail += 1
        shadow_vs_trail += 1


def _detect(ca, out, **kwargs):
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(shadow_vs_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    hi = ca.high
    lo = ca.low

    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_trail = start_idx - body_long_period

    shadow_vs_total = np.array([0.0, 0.0])
    body_long_total = np.array([0.0, 0.0])

    for j in range(shadow_vs_trail, start_idx):
        shadow_vs_total[1] += arr_svs[j - 1]
        shadow_vs_total[0] += arr_svs[j]

    for j in range(body_long_trail, start_idx):
        body_long_total[1] += arr_bl[j - 1]
        body_long_total[0] += arr_bl[j]

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.upper_shadow,
        ca.lower_shadow,
        hi,
        lo,
        arr_bl,
        arr_svs,
        shadow_vs_total,
        body_long_total,
        out,
        start_idx,
        shadow_vs_trail,
        body_long_trail,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.ShadowVeryShort],
    )


def cdl_kickingbylength(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Kickingbylength"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_KICKINGBYLENGTH",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
