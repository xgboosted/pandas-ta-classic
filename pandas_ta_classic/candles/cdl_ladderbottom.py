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
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 4
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = svs_period + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    svs_trail = start_idx - svs_period

    # Seed ShadowVeryShort total: applied to bar i-1
    svs_total = float(arr_svs[svs_trail - 1 : start_idx - 1].sum())
    O = ca.open
    H = ca.high
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # First three are black candlesticks
            ca.color[i - 4] == -1
            and ca.color[i - 3] == -1
            and ca.color[i - 2] == -1
            # With consecutively lower opens
            and O[i - 4] > O[i - 3]
            and O[i - 3] > O[i - 2]
            # And consecutively lower closes
            and C[i - 4] > C[i - 3]
            and C[i - 3] > C[i - 2]
            # 4th: black with an upper shadow
            and ca.color[i - 1] == -1
            and ca.upper_shadow[i - 1]
            > AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total
            # 5th: white
            and ca.color[i] == 1
            # That opens above prior candle's body (open, since bearish)
            and O[i] > O[i - 1]
            # And closes above prior candle's high
            and C[i] > H[i - 1]
        ):
            out[i] = 100

        # Update total: add current range, subtract trailing range
        svs_total += arr_svs[i - 1] - arr_svs[svs_trail - 1]
        svs_trail += 1


def cdl_ladderbottom(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Ladder Bottom

    A 5-candle bullish reversal pattern. Three consecutive black candles
    with lower opens and closes, followed by a black candle with a
    notable upper shadow, then a white candle that opens above the
    fourth candle's body and closes above its high.

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
        >>> result = cdl_ladderbottom(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_LADDERBOTTOM",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
