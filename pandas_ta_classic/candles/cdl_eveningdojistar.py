# Candle Evening Doji Star (CDL_EVENINGDOJISTAR)
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
    close,
    body_hi,
    body_lo,
    arr_bd,
    arr_bl,
    arr_bs,
    out,
    start_idx,
    body_long_trail,
    body_doji_trail,
    body_short_trail,
    body_long_total,
    body_doji_total,
    body_short_total,
    f_bl,
    f_bd,
    f_bs,
    penetration,
):
    for i in range(start_idx, len(out)):
        if (
            # 1st: long white
            real_body[i - 2] > f_bl * body_long_total
            and color[i - 2] == 1
            # 2nd: doji gapping up
            and real_body[i - 1] <= f_bd * body_doji_total
            and body_lo[i - 1] > body_hi[i - 2]
            # 3rd: longer than short, black, closing well within 1st rb
            and real_body[i] > f_bs * body_short_total
            and color[i] == -1
            and close[i] < close[i - 2] - real_body[i - 2] * penetration
        ):
            out[i] = -100

        # Update trailing windows
        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_doji_total += arr_bd[i - 1] - arr_bd[body_doji_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_doji_trail += 1
        body_short_trail += 1


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

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    # Trailing indices
    # BodyLong at offset i-2: trail = startIdx - 2 - period
    body_long_trail = start_idx - 2 - body_long_period
    # BodyDoji at offset i-1: trail = startIdx - 1 - period
    body_doji_trail = start_idx - 1 - body_doji_period
    # BodyShort at offset i: trail = startIdx - period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_doji_total = float(arr_bd[body_doji_trail : start_idx - 1].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.close,
        body_hi,
        body_lo,
        arr_bd,
        arr_bl,
        arr_bs,
        out,
        start_idx,
        body_long_trail,
        body_doji_trail,
        body_short_trail,
        body_long_total,
        body_doji_total,
        body_short_total,
        AVG_FACTOR[CandleSetting.BodyLong],
        AVG_FACTOR[CandleSetting.BodyDoji],
        AVG_FACTOR[CandleSetting.BodyShort],
        penetration,
    )


def cdl_eveningdojistar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: Optional[float] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Evening Doji Star

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
        "CDL_EVENINGDOJISTAR",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
