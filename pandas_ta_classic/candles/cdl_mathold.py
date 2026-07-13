# Candle Mat Hold (CDL_MATHOLD)
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
    real_body,
    color,
    O_,
    H,
    C,
    body_hi,
    body_lo,
    arr_bl,
    arr_bs,
    body_total,
    out,
    start_idx,
    body_short_trail,
    body_long_trail,
    f_bl,
    f_bs,
    penetration,
):
    for i in range(start_idx, len(out)):
        if (
            # 1st long, then 3 small
            real_body[i - 4] > f_bl * body_total[4]
            and real_body[i - 3] < f_bs * body_total[3]
            and real_body[i - 2] < f_bs * body_total[2]
            and real_body[i - 1] < f_bs * body_total[1]
            # white, black, ?, ?, white
            and color[i - 4] == 1
            and color[i - 3] == -1
            and color[i] == 1
            # upside gap 1st to 2nd
            and body_lo[i - 3] > body_hi[i - 4]
            # 3rd to 4th hold within 1st: part of real body within 1st body
            and min(O_[i - 2], C[i - 2]) < C[i - 4]
            and min(O_[i - 1], C[i - 1]) < C[i - 4]
            # reaction days penetrate first body less than penetration %
            and min(O_[i - 2], C[i - 2]) > C[i - 4] - real_body[i - 4] * penetration
            and min(O_[i - 1], C[i - 1]) > C[i - 4] - real_body[i - 4] * penetration
            # 2nd to 4th are falling
            and max(C[i - 2], O_[i - 2]) < O_[i - 3]
            and max(C[i - 1], O_[i - 1]) < max(C[i - 2], O_[i - 2])
            # 5th opens above the prior close
            and O_[i] > C[i - 1]
            # 5th closes above the highest high of the reaction days
            and C[i] > max(max(H[i - 3], H[i - 2]), H[i - 1])
        ):
            out[i] = 100  # Always bullish

        # Update totals
        body_total[4] += arr_bl[i - 4] - arr_bl[body_long_trail - 4]
        for k in range(3, 0, -1):
            body_total[k] += arr_bs[i - k] - arr_bs[body_short_trail - k]
        body_short_trail += 1
        body_long_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    penetration = kwargs.get("penetration", 0.5)

    # Lookback: max(TA_CANDLEAVGPERIOD(BodyShort), TA_CANDLEAVGPERIOD(BodyLong)) + 4
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_short_trail = start_idx - body_short_period
    body_long_trail = start_idx - body_long_period

    # Seed body totals: [0]=unused, [1..3]=BodyShort at i-1..i-3, [4]=BodyLong at i-4
    body_total = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Seed BodyShort totals for i-3, i-2, i-1
    j = body_short_trail
    while j < start_idx:
        body_total[3] += arr_bs[j - 3]
        body_total[2] += arr_bs[j - 2]
        body_total[1] += arr_bs[j - 1]
        j += 1

    # Seed BodyLong total for i-4
    j = body_long_trail
    while j < start_idx:
        body_total[4] += arr_bl[j - 4]
        j += 1

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.open,
        ca.high,
        ca.close,
        body_hi,
        body_lo,
        arr_bl,
        arr_bs,
        body_total,
        out,
        start_idx,
        body_short_trail,
        body_long_trail,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.BodyShort],
        penetration,
    )


def cdl_mathold(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: Optional[float] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Mat Hold

    A 5-candle bullish continuation pattern. Begins with a long white
    candle, followed by a gap-up and three small declining candles that
    stay within the first candle's body (penetrating no more than
    ``penetration`` percent), then a white candle that closes above the
    highest high of the reaction days.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        penetration: Maximum penetration of the first body by reaction
            days. Default: 0.5.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (bullish) / 0, or None.

    Example:
        >>> result = cdl_mathold(df.open, df.high, df.low, df.close, penetration=0.5)
    """
    if penetration is None:
        penetration = 0.5
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_MATHOLD",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
