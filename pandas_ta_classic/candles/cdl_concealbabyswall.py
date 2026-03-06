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
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 3
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = svs_period + 3
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    svs_trail = start_idx - svs_period

    # Seed totals for candles at i-3, i-2, i-1
    svs_total = [0.0, 0.0, 0.0, 0.0]  # indexed [0..3], use [1],[2],[3]
    for j in range(svs_trail, start_idx):
        svs_total[3] += arr_svs[j - 3]
        svs_total[2] += arr_svs[j - 2]
        svs_total[1] += arr_svs[j - 1]

    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # All four candles are black
            ca.color[i - 3] == -1
            and ca.color[i - 2] == -1
            and ca.color[i - 1] == -1
            and ca.color[i] == -1
            # 1st: marubozu (very short shadows)
            and ca.lower_shadow[i - 3]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total[3]
            and ca.upper_shadow[i - 3]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total[3]
            # 2nd: marubozu (very short shadows)
            and ca.lower_shadow[i - 2]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total[2]
            and ca.upper_shadow[i - 2]
            < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total[2]
            # 3rd: opens gapping down
            and body_hi[i - 1] < body_lo[i - 2]
            # 3rd: HAS an upper shadow
            and ca.upper_shadow[i - 1]
            > AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total[1]
            # 3rd upper shadow extends into the prior body
            and H[i - 1] > C[i - 2]
            # 4th: engulfs the 3rd including the shadows
            and H[i] > H[i - 1]
            and L[i] < L[i - 1]
        ):
            out[i] = 100  # Always bullish

        # Update totals
        for k in range(3, 0, -1):
            svs_total[k] += arr_svs[i - k] - arr_svs[svs_trail - k]
        svs_trail += 1


def cdl_concealbabyswall(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Concealing Baby Swallow

    A 4-candle bullish reversal pattern. All four candles are bearish.
    The first two are marubozu (very short shadows). The third opens
    gapping down but has an upper shadow that reaches into the second
    candle's body. The fourth completely engulfs the third (including
    shadows).

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (bullish) / 0, or None.

    Example:
        >>> result = cdl_concealbabyswall(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_CONCEALBABYSWALL",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
