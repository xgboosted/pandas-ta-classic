# Candle Advance Block (CDL_ADVANCEBLOCK)
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
def _detect_nb(color, real_body, upper_shadow, O_, C, ss_2, ss_1, ss_0, shadow_long, near_2, near_1, far_2, far_1, body_long_2, out, start_idx):
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
            # 2nd opens within/near 1st real body
            and O_[i - 1] > O_[i - 2]
            and O_[i - 1] <= C[i - 2] + near_2[i]
            # 3rd opens within/near 2nd real body
            and O_[i] > O_[i - 1]
            and O_[i] <= C[i - 1] + near_1[i]
            # 1st: long real body
            and real_body[i - 2] > body_long_2[i]
            # 1st: short upper shadow
            and upper_shadow[i - 2] < ss_2[i]
            # Signs of weakening (any of 4 sub-conditions)
            and (
                # Sub-condition 1: 2nd far smaller than 1st AND
                # 3rd not longer than 2nd
                (real_body[i - 1] < real_body[i - 2] - far_2[i] and real_body[i] < real_body[i - 1] + near_1[i])
                # Sub-condition 2: 3rd far smaller than 2nd
                or (real_body[i] < real_body[i - 1] - far_1[i])
                # Sub-condition 3: progressively smaller bodies AND
                # (3rd or 2nd has non-short upper shadow)
                or (
                    real_body[i] < real_body[i - 1]
                    and real_body[i - 1] < real_body[i - 2]
                    and (upper_shadow[i] > ss_0[i] or upper_shadow[i - 1] > ss_1[i])
                )
                # Sub-condition 4: 3rd smaller than 2nd AND
                # 3rd has long upper shadow
                or (real_body[i] < real_body[i - 1] and upper_shadow[i] > shadow_long[i])
            )
        ):
            out[i] = -100  # Always bearish


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    settings = (CandleSetting.ShadowLong, CandleSetting.ShadowShort, CandleSetting.Far, CandleSetting.Near, CandleSetting.BodyLong)
    # Lookback: max(all avg periods) + 2
    start_idx = max(candle_avg_period(s) for s in settings) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.upper_shadow,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.ShadowShort, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowShort, 0, start_idx),
        candle_average(ca, CandleSetting.ShadowLong, 0, start_idx),
        candle_average(ca, CandleSetting.Near, 2, start_idx),
        candle_average(ca, CandleSetting.Near, 1, start_idx),
        candle_average(ca, CandleSetting.Far, 2, start_idx),
        candle_average(ca, CandleSetting.Far, 1, start_idx),
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        out,
        start_idx,
    )


def cdl_advanceblock(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Advance Block

    A 3-candle bearish reversal pattern. Three consecutive white candles
    with consecutively higher closes, each opening within or near the
    previous real body. The first has a long body with a short upper
    shadow. Signs of weakening appear: progressively smaller real bodies,
    relatively long upper shadows, or the second/third candle being far
    shorter than the prior one.

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
        >>> result = cdl_advanceblock(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ADVANCEBLOCK",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
