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

    # Lookback: max(BodyDoji, BodyLong, BodyShort) + 2
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_doji_period, body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    # Trailing indices: each setting has its own trail
    # BodyLong at offset i-2: trail = startIdx - 2 - period
    body_long_trail = start_idx - 2 - body_long_period
    # BodyDoji at offset i-1: trail = startIdx - 1 - period
    body_doji_trail = start_idx - 1 - body_doji_period
    # BodyShort at offset i: trail = startIdx - period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    body_long_total = 0.0
    for j in range(body_long_trail, start_idx - 2):
        body_long_total += ca.candle_range(CandleSetting.BodyLong, j)

    body_doji_total = 0.0
    for j in range(body_doji_trail, start_idx - 1):
        body_doji_total += ca.candle_range(CandleSetting.BodyDoji, j)

    body_short_total = 0.0
    for j in range(body_short_trail, start_idx):
        body_short_total += ca.candle_range(CandleSetting.BodyShort, j)

    for i in range(start_idx, len(out)):
        # Pattern detection
        if (
            # 1st: long real body
            ca.real_body[i - 2]
            > ca.candle_average(CandleSetting.BodyLong, body_long_total, i - 2)
            # 2nd: doji
            and ca.real_body[i - 1]
            <= ca.candle_average(CandleSetting.BodyDoji, body_doji_total, i - 1)
            # 3rd: longer than short
            and ca.real_body[i]
            > ca.candle_average(CandleSetting.BodyShort, body_short_total, i)
            and (
                (
                    # Bullish 1st white, bearish 3rd black
                    ca.color[i - 2] == 1
                    and ca.color[i] == -1
                    # 3rd closes well within 1st rb
                    and ca.close[i]
                    < ca.close[i - 2] - ca.real_body[i - 2] * penetration
                    # upside candle gap between 1st and 2nd
                    and ca.candle_gap_up(i - 1, i - 2)
                    # downside candle gap between 2nd and 3rd
                    and ca.candle_gap_down(i, i - 1)
                )
                or (
                    # Bearish 1st black, bullish 3rd white
                    ca.color[i - 2] == -1
                    and ca.color[i] == 1
                    # 3rd closes well within 1st rb
                    and ca.close[i]
                    > ca.close[i - 2] + ca.real_body[i - 2] * penetration
                    # downside candle gap between 1st and 2nd
                    and ca.candle_gap_down(i - 1, i - 2)
                    # upside candle gap between 2nd and 3rd
                    and ca.candle_gap_up(i, i - 1)
                )
            )
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_long_total += ca.candle_range(
            CandleSetting.BodyLong, i - 2
        ) - ca.candle_range(CandleSetting.BodyLong, body_long_trail)
        body_doji_total += ca.candle_range(
            CandleSetting.BodyDoji, i - 1
        ) - ca.candle_range(CandleSetting.BodyDoji, body_doji_trail)
        body_short_total += ca.candle_range(
            CandleSetting.BodyShort, i
        ) - ca.candle_range(CandleSetting.BodyShort, body_short_trail)
        body_long_trail += 1
        body_doji_trail += 1
        body_short_trail += 1


def cdl_abandonedbaby(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: Optional[float] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Abandoned Baby

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
        A pandas Series with +100 (bullish) or -100 (bearish) or 0.
    """
    if penetration is None:
        penetration = 0.3
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ABANDONEDBABY",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
