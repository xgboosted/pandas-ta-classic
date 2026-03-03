# -*- coding: utf-8 -*-
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)
import numpy as np


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Settings and their avg periods
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    shadow_vshort_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)

    # Lookback: max(all avg periods) + 2
    lookback = (
        max(
            shadow_vshort_period,
            shadow_long_period,
            body_long_period,
            body_short_period,
        )
        + 2
    )
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices
    body_long_trail = start_idx - body_long_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vshort_trail = start_idx - shadow_vshort_period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    # BodyLong: applied to i-2
    body_long_total = 0.0
    for j in range(body_long_trail, start_idx):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j - 2)

    # ShadowLong: applied to i-2
    shadow_long_total = 0.0
    for j in range(shadow_long_trail, start_idx):
        shadow_long_total += ca.candle_range(CandleSetting.ShadowLong, j - 2)

    # ShadowVeryShort[1]: applied to i-1; ShadowVeryShort[0]: applied to i
    shadow_vshort_total_1 = 0.0
    shadow_vshort_total_0 = 0.0
    for j in range(shadow_vshort_trail, start_idx):
        shadow_vshort_total_1 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 1)
        shadow_vshort_total_0 += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    # BodyShort: applied to i
    body_short_total = 0.0
    for j in range(body_short_trail, start_idx):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    O = ca.open
    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # All three candles are black
            ca.color[i - 2] == -1
            and ca.color[i - 1] == -1
            and ca.color[i] == -1
            # 1st: long body
            and ca.real_body[i - 2]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 2)
            # 1st: long lower shadow
            and ca.lower_shadow[i - 2]
            > ca.candle_average(CandleSetting.ShadowLong, shadow_long_total, i - 2)
            # 2nd: smaller candle
            and ca.real_body[i - 1] < ca.real_body[i - 2]
            # 2nd: opens higher than 1st close but within 1st range
            and O[i - 1] > C[i - 2]
            and O[i - 1] <= H[i - 2]
            # 2nd: trades lower than 1st close
            and L[i - 1] < C[i - 2]
            # 2nd: but not lower than 1st low
            and L[i - 1] >= L[i - 2]
            # 2nd: has a lower shadow (not very short)
            and ca.lower_shadow[i - 1]
            > ca.candle_average(
                CandleSetting.ShadowVeryShort, shadow_vshort_total_1, i - 1
            )
            # 3rd: small marubozu (short body)
            and ca.real_body[i]
            < ca.candle_average(CandleSetting.BodyShort, body_short_total, i)
            # 3rd: very short lower shadow
            and ca.lower_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, shadow_vshort_total_0, i)
            # 3rd: very short upper shadow
            and ca.upper_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, shadow_vshort_total_0, i)
            # 3rd: engulfed by 2nd candle's range
            and L[i] > L[i - 1]
            and H[i] < H[i - 1]
        ):
            out[i] = 100  # Always bullish

        # Update totals
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 2
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - 2)
        shadow_long_total += ca.candle_range(
            CandleSetting.ShadowLong, i - 2
        ) - ca.candle_range(CandleSetting.ShadowLong, shadow_long_trail - 2)

        shadow_vshort_total_1 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 1
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vshort_trail - 1)
        shadow_vshort_total_0 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, shadow_vshort_trail)

        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)

        body_long_trail += 1
        shadow_long_trail += 1
        shadow_vshort_trail += 1
        body_short_trail += 1


def cdl_3starsinsouth(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
