# Candle Three Stars In The South (CDL_3STARSINSOUTH)
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
def _detect_nb(color, real_body, lower_shadow, upper_shadow, O_, H, L, C, body_long_2, shadow_long_2, svs_1, svs_0, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # All three candles are black
            color[i - 2] == -1
            and color[i - 1] == -1
            and color[i] == -1
            # 1st: long body
            and real_body[i - 2] > body_long_2[i]
            # 1st: long lower shadow
            and lower_shadow[i - 2] > shadow_long_2[i]
            # 2nd: smaller candle
            and real_body[i - 1] < real_body[i - 2]
            # 2nd: opens higher than 1st close but within 1st range
            and O_[i - 1] > C[i - 2]
            and O_[i - 1] <= H[i - 2]
            # 2nd: trades lower than 1st close
            and L[i - 1] < C[i - 2]
            # 2nd: but not lower than 1st low
            and L[i - 1] >= L[i - 2]
            # 2nd: has a lower shadow (not very short)
            and lower_shadow[i - 1] > svs_1[i]
            # 3rd: small marubozu (short body)
            and real_body[i] < body_short[i]
            # 3rd: very short lower shadow
            and lower_shadow[i] < svs_0[i]
            # 3rd: very short upper shadow
            and upper_shadow[i] < svs_0[i]
            # 3rd: engulfed by 2nd candle's range
            and L[i] > L[i - 1]
            and H[i] < H[i - 1]
        ):
            out[i] = 100  # Always bullish


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    settings = (CandleSetting.ShadowVeryShort, CandleSetting.ShadowLong, CandleSetting.BodyLong, CandleSetting.BodyShort)
    # Lookback: max(all avg periods) + 2
    start_idx = max(candle_avg_period(s) for s in settings) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        ca.open,
        ca.high,
        ca.low,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_3starsinsouth(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Three Stars In The South

    A 3-candle bullish reversal pattern. All three candles are bearish.
    The first is a long black candle with a long lower shadow. The second
    is a smaller black candle that opens higher than the first's close but
    within the first's range, trades lower than the first's close but not
    lower than its low, and has a lower shadow. The third is a small black
    marubozu engulfed by the second candle's range.

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
        >>> result = cdl_3starsinsouth(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3STARSINSOUTH",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
