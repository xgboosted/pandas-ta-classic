# Candle Mat Hold (CDL_MATHOLD)
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
from pandas_ta_classic.utils._core import _number
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(real_body, color, O_, H, C, body_hi, body_lo, body_long_4, body_short_3, body_short_2, body_short_1, out, start_idx, penetration):
    for i in range(start_idx, len(out)):
        if (
            # 1st long, then 3 small
            real_body[i - 4] > body_long_4[i]
            and real_body[i - 3] < body_short_3[i]
            and real_body[i - 2] < body_short_2[i]
            and real_body[i - 1] < body_short_1[i]
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


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    penetration = kwargs["penetration"]

    # Lookback: max(TA_CANDLEAVGPERIOD(BodyShort), TA_CANDLEAVGPERIOD(BodyLong)) + 4
    start_idx = max(candle_avg_period(CandleSetting.BodyShort), candle_avg_period(CandleSetting.BodyLong)) + 4
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.open,
        ca.high,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyLong, 4, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 3, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 2, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 1, start_idx, sequential_seed=True),
        out,
        start_idx,
        penetration,
    )


def cdl_mathold(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: float | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
    # TA-Lib rejects a negative penetration (TA_BAD_PARAM)
    penetration = _number(penetration, 0.5, "penetration", ge=0)
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
