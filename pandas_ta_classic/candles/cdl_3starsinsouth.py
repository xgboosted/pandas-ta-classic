# Candle Three Stars In The South (CDL_3STARSINSOUTH)
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
    color,
    real_body,
    lower_shadow,
    upper_shadow,
    O_,
    H,
    L,
    C,
    arr_bl,
    arr_bs,
    arr_sl,
    arr_svs,
    out,
    start_idx,
    body_long_trail,
    shadow_long_trail,
    shadow_vshort_trail,
    body_short_trail,
    body_long_total,
    shadow_long_total,
    shadow_vshort_total_1,
    shadow_vshort_total_0,
    body_short_total,
    f_bl,
    f_sl,
    f_svs,
    f_bs,
):
    for i in range(start_idx, len(out)):
        if (
            # All three candles are black
            color[i - 2] == -1
            and color[i - 1] == -1
            and color[i] == -1
            # 1st: long body
            and real_body[i - 2] > f_bl * body_long_total
            # 1st: long lower shadow
            and lower_shadow[i - 2] > f_sl * arr_sl[i - 2]
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
            and lower_shadow[i - 1] > f_svs * shadow_vshort_total_1
            # 3rd: small marubozu (short body)
            and real_body[i] < f_bs * body_short_total
            # 3rd: very short lower shadow
            and lower_shadow[i] < f_svs * shadow_vshort_total_0
            # 3rd: very short upper shadow
            and upper_shadow[i] < f_svs * shadow_vshort_total_0
            # 3rd: engulfed by 2nd candle's range
            and L[i] > L[i - 1]
            and H[i] < H[i - 1]
        ):
            out[i] = 100  # Always bullish

        # Update totals
        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail - 2]
        shadow_long_total += arr_sl[i - 2] - arr_sl[shadow_long_trail - 2]

        shadow_vshort_total_1 += arr_svs[i - 1] - arr_svs[shadow_vshort_trail - 1]
        shadow_vshort_total_0 += arr_svs[i] - arr_svs[shadow_vshort_trail]

        body_short_total += arr_bs[i] - arr_bs[body_short_trail]

        body_long_trail += 1
        shadow_long_trail += 1
        shadow_vshort_trail += 1
        body_short_trail += 1


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

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    # Trailing indices
    body_long_trail = start_idx - body_long_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vshort_trail = start_idx - shadow_vshort_period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    # BodyLong: applied to i-2
    body_long_total = float(arr_bl[body_long_trail - 2 : start_idx - 2].sum())
    # ShadowLong: applied to i-2
    shadow_long_total = float(arr_sl[shadow_long_trail - 2 : start_idx - 2].sum())
    # ShadowVeryShort[1]: applied to i-1; ShadowVeryShort[0]: applied to i
    shadow_vshort_total_1 = float(arr_svs[shadow_vshort_trail - 1 : start_idx - 1].sum())
    shadow_vshort_total_0 = float(arr_svs[shadow_vshort_trail:start_idx].sum())
    # BodyShort: applied to i
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.lower_shadow,
        ca.upper_shadow,
        ca.open,
        ca.high,
        ca.low,
        ca.close,
        arr_bl,
        arr_bs,
        arr_sl,
        arr_svs,
        out,
        start_idx,
        body_long_trail,
        shadow_long_trail,
        shadow_vshort_trail,
        body_short_trail,
        body_long_total,
        shadow_long_total,
        shadow_vshort_total_1,
        shadow_vshort_total_0,
        body_short_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.ShadowLong],
        AVG_FACTOR[CandleSetting.ShadowVeryShort],
        AVG_FACTOR[CandleSetting.BodyShort],
    )


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
