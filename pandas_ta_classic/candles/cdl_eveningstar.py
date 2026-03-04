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
    penetration = kwargs.get("penetration", 0.3)

    # Lookback: max(BodyShort, BodyLong) + 2
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_short_period, body_long_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices
    # BodyLong at offset i-2: trail = startIdx - 2 - period
    body_long_trail = start_idx - 2 - body_long_period
    # BodyShort at offset i-1: trail = startIdx - 1 - period
    # Both BodyShort totals share the same trailing index
    body_short_trail = start_idx - 1 - body_short_period

    # Seed totals
    body_long_total = 0.0
    for j in range(body_long_trail, start_idx - 2):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    # BodyShortPeriodTotal tracks BodyShort at i-1
    # BodyShortPeriodTotal2 tracks BodyShort at i
    body_short_total = 0.0
    body_short_total2 = 0.0
    for j in range(body_short_trail, start_idx - 1):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)
        body_short_total2 += ca.candle_range(CandleSetting.BodyShort, j + 1)

    for i in range(start_idx, len(out)):
        if (
            # 1st: long white
            ca.real_body[i - 2]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 2)
            and ca.color[i - 2] == 1
            # 2nd: short, gapping up
            and ca.real_body[i - 1]
            <= ca.candle_average(CandleSetting.BodyShort, body_short_total, i - 1)
            and ca.real_body_gap_up(i - 1, i - 2)
            # 3rd: longer than short, black, closing well within 1st rb
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyShort, body_short_total2, i)
            and ca.color[i] == -1
            and ca.close[i] < ca.close[i - 2] - ca.real_body[i - 2] * penetration
        ):
            out[i] = -100

        # Update trailing windows
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 2
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i - 1
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)
        body_short_total2 += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail + 1)
        body_long_trail += 1
        body_short_trail += 1


def cdl_eveningstar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: Optional[float] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Evening Star

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        penetration: Percentage of penetration within the first candle's
            real body. Default: 0.3
        scalar: Multiplier for output values. Default: 100.
        offset: How many periods to shift the result.

    Returns:
        A pandas Series with -100 (bearish) or 0.
    """
    if penetration is None:
        penetration = 0.3
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_EVENINGSTAR",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
