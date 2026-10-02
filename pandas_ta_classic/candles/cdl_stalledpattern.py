# Candle Stalled Pattern (CDL_STALLEDPATTERN)
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
def _detect_nb(color, real_body, upper_shadow, O_, C, body_long_2, body_long_1, body_short, svs_1, near_2, near_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # 1st white
            color[i - 2] == 1
            # 2nd white
            and color[i - 1] == 1
            # 3rd white
            and color[i] == 1
            # Consecutive higher closes
            and C[i] > C[i - 1]
            and C[i - 1] > C[i - 2]
            # 1st: long real body
            and real_body[i - 2] > body_long_2[i]
            # 2nd: long real body
            and real_body[i - 1] > body_long_1[i]
            # 2nd: very short upper shadow
            and upper_shadow[i - 1] < svs_1[i]
            # 2nd opens within/near 1st real body: opens above 1st open
            and O_[i - 1] > O_[i - 2]
            # 2nd opens at or below 1st close + Near average
            and O_[i - 1] <= C[i - 2] + near_2[i]
            # 3rd: small real body
            and real_body[i] < body_short[i]
            # 3rd rides on the shoulder of 2nd real body
            and O_[i] >= C[i - 1] - real_body[i] - near_1[i]
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(max(BodyLong, BodyShort),
    #               max(ShadowVeryShort, Near)) + 2
    settings = (CandleSetting.BodyLong, CandleSetting.BodyShort, CandleSetting.ShadowVeryShort, CandleSetting.Near)
    start_idx = max(candle_avg_period(s) for s in settings) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.upper_shadow,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.Near, 2, start_idx),
        candle_average(ca, CandleSetting.Near, 1, start_idx),
        out,
        start_idx,
    )


def cdl_stalledpattern(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Stalled Pattern

    Three white candlesticks with consecutively higher closes. The first
    two have long real bodies; the second has a very short upper shadow
    and opens within or near the first's real body. The third has a
    small real body that gaps away or rides on the shoulder of the
    second's body, signaling a potential reversal.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_stalledpattern(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_STALLEDPATTERN",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
