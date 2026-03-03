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
    shadow_short_period = candle_avg_period(CandleSetting.ShadowShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    near_period = candle_avg_period(CandleSetting.Near)
    far_period = candle_avg_period(CandleSetting.Far)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)

    # Lookback: max(all avg periods) + 2
    lookback = (
        max(
            shadow_long_period,
            shadow_short_period,
            far_period,
            near_period,
            body_long_period,
        )
        + 2
    )
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices
    shadow_short_trail = start_idx - shadow_short_period
    shadow_long_trail = start_idx - shadow_long_period
    near_trail = start_idx - near_period
    far_trail = start_idx - far_period
    body_long_trail = start_idx - body_long_period

    # Seed ShadowShort totals [2], [1], [0]
    ss_total_2 = 0.0
    ss_total_1 = 0.0
    ss_total_0 = 0.0
    for j in range(shadow_short_trail, start_idx):
        ss_total_2 += ca.candle_range(CandleSetting.ShadowShort, j - 2)
        ss_total_1 += ca.candle_range(CandleSetting.ShadowShort, j - 1)
        ss_total_0 += ca.candle_range(CandleSetting.ShadowShort, j)

    # Seed ShadowLong totals [1], [0]
    sl_total_1 = 0.0
    sl_total_0 = 0.0
    for j in range(shadow_long_trail, start_idx):
        sl_total_1 += ca.candle_range(CandleSetting.ShadowLong, j - 1)
        sl_total_0 += ca.candle_range(CandleSetting.ShadowLong, j)

    # Seed Near totals [2], [1]
    near_total_2 = 0.0
    near_total_1 = 0.0
    for j in range(near_trail, start_idx):
        near_total_2 += ca.candle_range(CandleSetting.Near, j - 2)
        near_total_1 += ca.candle_range(CandleSetting.Near, j - 1)

    # Seed Far totals [2], [1]
    far_total_2 = 0.0
    far_total_1 = 0.0
    for j in range(far_trail, start_idx):
        far_total_2 += ca.candle_range(CandleSetting.Far, j - 2)
        far_total_1 += ca.candle_range(CandleSetting.Far, j - 1)

    # Seed BodyLong total (applied to i-2)
    body_long_total = 0.0
    for j in range(body_long_trail, start_idx):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j - 2)

    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st white
            ca.color[i - 2] == 1
            # 2nd white
            and ca.color[i - 1] == 1
            # 3rd white
            and ca.color[i] == 1
            # Consecutive higher closes
            and C[i] > C[i - 1]
            and C[i - 1] > C[i - 2]
            # 2nd opens within/near 1st real body
            and O[i - 1] > O[i - 2]
            and O[i - 1]
            <= C[i - 2] + ca.candle_average(CandleSetting.Near, near_total_2, i - 2)
            # 3rd opens within/near 2nd real body
            and O[i] > O[i - 1]
            and O[i]
            <= C[i - 1] + ca.candle_average(CandleSetting.Near, near_total_1, i - 1)
            # 1st: long real body
            and ca.real_body[i - 2]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 2)
            # 1st: short upper shadow
            and ca.upper_shadow[i - 2]
            < ca.candle_average(CandleSetting.ShadowShort, ss_total_2, i - 2)
            # Signs of weakening (any of 4 sub-conditions)
            and (
                # Sub-condition 1: 2nd far smaller than 1st AND
                # 3rd not longer than 2nd
                (
                    ca.real_body[i - 1]
                    < ca.real_body[i - 2]
                    - ca.candle_average(CandleSetting.Far, far_total_2, i - 2)
                    and ca.real_body[i]
                    < ca.real_body[i - 1]
                    + ca.candle_average(CandleSetting.Near, near_total_1, i - 1)
                )
                # Sub-condition 2: 3rd far smaller than 2nd
                or (
                    ca.real_body[i]
                    < ca.real_body[i - 1]
                    - ca.candle_average(CandleSetting.Far, far_total_1, i - 1)
                )
                # Sub-condition 3: progressively smaller bodies AND
                # (3rd or 2nd has non-short upper shadow)
                or (
                    ca.real_body[i] < ca.real_body[i - 1]
                    and ca.real_body[i - 1] < ca.real_body[i - 2]
                    and (
                        ca.upper_shadow[i]
                        > ca.candle_average(CandleSetting.ShadowShort, ss_total_0, i)
                        or ca.upper_shadow[i - 1]
                        > ca.candle_average(
                            CandleSetting.ShadowShort, ss_total_1, i - 1
                        )
                    )
                )
                # Sub-condition 4: 3rd smaller than 2nd AND
                # 3rd has long upper shadow
                or (
                    ca.real_body[i] < ca.real_body[i - 1]
                    and ca.upper_shadow[i]
                    > ca.candle_average(CandleSetting.ShadowLong, sl_total_0, i)
                )
            )
        ):
            out[i] = -100  # Always bearish

        # Update ShadowShort totals [2], [1], [0]
        ss_total_2 += ca.candle_range(
            CandleSetting.ShadowShort, i - 2
        ) - ca.candle_range(CandleSetting.ShadowShort, shadow_short_trail - 2)
        ss_total_1 += ca.candle_range(
            CandleSetting.ShadowShort, i - 1
        ) - ca.candle_range(CandleSetting.ShadowShort, shadow_short_trail - 1)
        ss_total_0 += ca.candle_range(CandleSetting.ShadowShort, i) - ca.candle_range(
            CandleSetting.ShadowShort, shadow_short_trail
        )

        # Update ShadowLong totals [1], [0]
        sl_total_1 += ca.candle_range(
            CandleSetting.ShadowLong, i - 1
        ) - ca.candle_range(CandleSetting.ShadowLong, shadow_long_trail - 1)
        sl_total_0 += ca.candle_range(CandleSetting.ShadowLong, i) - ca.candle_range(
            CandleSetting.ShadowLong, shadow_long_trail
        )

        # Update Far and Near totals [2], [1]
        far_total_2 += ca.candle_range(CandleSetting.Far, i - 2) - ca.candle_range(
            CandleSetting.Far, far_trail - 2
        )
        far_total_1 += ca.candle_range(CandleSetting.Far, i - 1) - ca.candle_range(
            CandleSetting.Far, far_trail - 1
        )
        near_total_2 += ca.candle_range(CandleSetting.Near, i - 2) - ca.candle_range(
            CandleSetting.Near, near_trail - 2
        )
        near_total_1 += ca.candle_range(CandleSetting.Near, i - 1) - ca.candle_range(
            CandleSetting.Near, near_trail - 1
        )

        # Update BodyLong total (applied to i-2)
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 2
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail - 2)

        shadow_short_trail += 1
        shadow_long_trail += 1
        near_trail += 1
        far_trail += 1
        body_long_trail += 1


def cdl_advanceblock(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
