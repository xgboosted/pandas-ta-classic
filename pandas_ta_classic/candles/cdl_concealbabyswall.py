# Candle Concealing Baby Swallow (CDL_CONCEALBABYSWALL)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    CandleSetting,
    candle_average,
    candle_avg_period,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(color, lower_shadow, upper_shadow, H, L, C, body_hi, body_lo, svs_3, svs_2, svs_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # All four candles are black
            color[i - 3] == -1
            and color[i - 2] == -1
            and color[i - 1] == -1
            and color[i] == -1
            # 1st: marubozu (very short shadows)
            and lower_shadow[i - 3] < svs_3[i]
            and upper_shadow[i - 3] < svs_3[i]
            # 2nd: marubozu (very short shadows)
            and lower_shadow[i - 2] < svs_2[i]
            and upper_shadow[i - 2] < svs_2[i]
            # 3rd: opens gapping down
            and body_hi[i - 1] < body_lo[i - 2]
            # 3rd: HAS an upper shadow
            and upper_shadow[i - 1] > svs_1[i]
            # 3rd upper shadow extends into the prior body
            and H[i - 1] > C[i - 2]
            # 4th: engulfs the 3rd including the shadows
            and H[i] > H[i - 1]
            and L[i] < L[i - 1]
        ):
            out[i] = 100  # Always bullish


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 3
    start_idx = candle_avg_period(CandleSetting.ShadowVeryShort) + 3
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.lower_shadow,
        ca.upper_shadow,
        ca.high,
        ca.low,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.ShadowVeryShort, 3, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.ShadowVeryShort, 2, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx, sequential_seed=True),
        out,
        start_idx,
    )


def cdl_concealbabyswall(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
