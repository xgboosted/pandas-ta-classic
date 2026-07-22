# Candle Breakaway (CDL_BREAKAWAY)
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
    O_,
    H,
    L,
    C,
    body_hi,
    body_lo,
    arr_bl,
    out,
    start_idx,
    body_long_trail,
    body_long_total,
    f_bl,
):
    for i in range(start_idx, len(out)):
        if (
            # 1st: long body
            real_body[i - 4] > f_bl * body_long_total
            # 1st, 2nd, 4th same color; 5th opposite
            and color[i - 4] == color[i - 3]
            and color[i - 3] == color[i - 1]
            and color[i - 1] == -color[i]
            and (
                (
                    # When 1st is black:
                    color[i - 4] == -1
                    # 2nd gaps down
                    and body_hi[i - 3] < body_lo[i - 4]
                    # 3rd has lower high and low than 2nd
                    and H[i - 2] < H[i - 3]
                    and L[i - 2] < L[i - 3]
                    # 4th has lower high and low than 3rd
                    and H[i - 1] < H[i - 2]
                    and L[i - 1] < L[i - 2]
                    # 5th closes inside the gap
                    and C[i] > O_[i - 3]
                    and C[i] < C[i - 4]
                )
                or (
                    # When 1st is white:
                    color[i - 4] == 1
                    # 2nd gaps up
                    and body_lo[i - 3] > body_hi[i - 4]
                    # 3rd has higher high and low than 2nd
                    and H[i - 2] > H[i - 3]
                    and L[i - 2] > L[i - 3]
                    # 4th has higher high and low than 3rd
                    and H[i - 1] > H[i - 2]
                    and L[i - 1] > L[i - 2]
                    # 5th closes inside the gap
                    and C[i] < O_[i - 3]
                    and C[i] > C[i - 4]
                )
            )
        ):
            out[i] = color[i] * 100

        # Update: add current, subtract trailing (both reference i-4)
        body_long_total += arr_bl[i - 4] - arr_bl[body_long_trail - 4]
        body_long_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 4
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    # Trailing index for BodyLong setting applied to i-4
    body_long_trail = start_idx - body_long_period

    # Seed BodyLong total: sum of the BodyLong range values at i-4
    # for i from body_long_trail to start_idx-1
    body_long_total = float(arr_bl[body_long_trail - 4 : start_idx - 4].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.high,
        ca.low,
        ca.close,
        body_hi,
        body_lo,
        arr_bl,
        out,
        start_idx,
        body_long_trail,
        body_long_total,
        AVG_FACTOR[CandleSetting.BodyLong],
    )


def cdl_breakaway(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Breakaway

    A 5-candle reversal pattern. Bullish breakaway begins with a long
    bearish candle, followed by a gap-down and two more bearish candles
    with successively lower highs/lows, then a bullish candle that
    closes inside the initial gap. Bearish breakaway is the mirror.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (bullish) / -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_breakaway(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_BREAKAWAY",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
