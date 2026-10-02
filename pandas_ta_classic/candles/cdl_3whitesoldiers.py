# Candle Three Advancing White Soldiers (CDL_3WHITESOLDIERS)
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
def _detect_nb(color, upper_shadow, real_body, O_, C, svs_2, svs_1, svs_0, near_2, near_1, far_2, far_1, body_short, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # 1st white
            color[i - 2] == 1
            # 1st: very short upper shadow
            and upper_shadow[i - 2] < svs_2[i]
            # 2nd white
            and color[i - 1] == 1
            # 2nd: very short upper shadow
            and upper_shadow[i - 1] < svs_1[i]
            # 3rd white
            and color[i] == 1
            # 3rd: very short upper shadow
            and upper_shadow[i] < svs_0[i]
            # Consecutive higher closes
            and C[i] > C[i - 1]
            and C[i - 1] > C[i - 2]
            # 2nd opens within/near 1st real body
            and O_[i - 1] > O_[i - 2]
            and O_[i - 1] <= C[i - 2] + near_2[i]
            # 3rd opens within/near 2nd real body
            and O_[i] > O_[i - 1]
            and O_[i] <= C[i - 1] + near_1[i]
            # 2nd not far shorter than 1st
            and real_body[i - 1] > real_body[i - 2] - far_2[i]
            # 3rd not far shorter than 2nd
            and real_body[i] > real_body[i - 1] - far_1[i]
            # 3rd: not short real body
            and real_body[i] > body_short[i]
        ):
            out[i] = 100  # Always bullish


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.ShadowVeryShort, CandleSetting.BodyShort, CandleSetting.Far, CandleSetting.Near)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.upper_shadow,
        ca.real_body,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.ShadowVeryShort, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        candle_average(ca, CandleSetting.Near, 2, start_idx),
        candle_average(ca, CandleSetting.Near, 1, start_idx),
        candle_average(ca, CandleSetting.Far, 2, start_idx),
        candle_average(ca, CandleSetting.Far, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_3whitesoldiers(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Three Advancing White Soldiers

    A 3-candle bullish reversal pattern. Three consecutive white
    (bullish) candles with consecutively higher closes. Each candle
    opens within or near the previous real body and has very short
    upper shadows. Each candle must not be far shorter than the prior
    one, and the third must not be short.

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
        >>> result = cdl_3whitesoldiers(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3WHITESOLDIERS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
