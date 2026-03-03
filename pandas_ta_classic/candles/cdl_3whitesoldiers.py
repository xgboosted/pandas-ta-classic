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
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    far_period = candle_avg_period(CandleSetting.Far)
    near_period = candle_avg_period(CandleSetting.Near)

    # Lookback: max(all avg periods) + 2
    lookback = max(svs_period, body_short_period, far_period, near_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices
    svs_trail = start_idx - svs_period
    near_trail = start_idx - near_period
    far_trail = start_idx - far_period
    body_short_trail = start_idx - body_short_period

    # Seed ShadowVeryShort totals [2], [1], [0]
    svs_total_2 = 0.0
    svs_total_1 = 0.0
    svs_total_0 = 0.0
    for j in range(svs_trail, start_idx):
        svs_total_2 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 2)
        svs_total_1 += ca.candle_range(CandleSetting.ShadowVeryShort, j - 1)
        svs_total_0 += ca.candle_range(CandleSetting.ShadowVeryShort, j)

    # Seed Near totals [2], [1] (not [0])
    near_total_2 = 0.0
    near_total_1 = 0.0
    for j in range(near_trail, start_idx):
        near_total_2 += ca.candle_range(CandleSetting.Near, j - 2)
        near_total_1 += ca.candle_range(CandleSetting.Near, j - 1)

    # Seed Far totals [2], [1] (not [0])
    far_total_2 = 0.0
    far_total_1 = 0.0
    for j in range(far_trail, start_idx):
        far_total_2 += ca.candle_range(CandleSetting.Far, j - 2)
        far_total_1 += ca.candle_range(CandleSetting.Far, j - 1)

    # Seed BodyShort total
    body_short_total = 0.0
    for j in range(body_short_trail, start_idx):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st white
            ca.color[i - 2] == 1
            # 1st: very short upper shadow
            and ca.upper_shadow[i - 2]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_2, i - 2)
            # 2nd white
            and ca.color[i - 1] == 1
            # 2nd: very short upper shadow
            and ca.upper_shadow[i - 1]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_1, i - 1)
            # 3rd white
            and ca.color[i] == 1
            # 3rd: very short upper shadow
            and ca.upper_shadow[i]
            < ca.candle_average(CandleSetting.ShadowVeryShort, svs_total_0, i)
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
            # 2nd not far shorter than 1st
            and ca.real_body[i - 1]
            > ca.real_body[i - 2]
            - ca.candle_average(CandleSetting.Far, far_total_2, i - 2)
            # 3rd not far shorter than 2nd
            and ca.real_body[i]
            > ca.real_body[i - 1]
            - ca.candle_average(CandleSetting.Far, far_total_1, i - 1)
            # 3rd: not short real body
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyShort, body_short_total, i)
        ):
            out[i] = 100  # Always bullish

        # Update ShadowVeryShort totals [2], [1], [0]
        svs_total_2 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 2
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 2)
        svs_total_1 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i - 1
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail - 1)
        svs_total_0 += ca.candle_range(
            CandleSetting.ShadowVeryShort, i
        ) - ca.candle_range(CandleSetting.ShadowVeryShort, svs_trail)

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

        # Update BodyShort total
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)

        svs_trail += 1
        near_trail += 1
        far_trail += 1
        body_short_trail += 1


def cdl_3whitesoldiers(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
