# Candle Evening Doji Star (CDL_EVENINGDOJISTAR)
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
def _detect_nb(real_body, color, close, body_hi, body_lo, body_long_2, body_doji_1, body_short_0, out, start_idx, penetration):
    for i in range(start_idx, len(out)):
        if (
            # 1st: long white
            real_body[i - 2] > body_long_2[i]
            and color[i - 2] == 1
            # 2nd: doji gapping up
            and real_body[i - 1] <= body_doji_1[i]
            and body_lo[i - 1] > body_hi[i - 2]
            # 3rd: longer than short, black, closing well within 1st rb
            and real_body[i] > body_short_0[i]
            and color[i] == -1
            and close[i] < close[i - 2] - real_body[i - 2] * penetration
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    penetration = kwargs["penetration"]

    # Lookback: max(BodyDoji, BodyLong, BodyShort) + 2
    start_idx = max(candle_avg_period(s) for s in (CandleSetting.BodyDoji, CandleSetting.BodyLong, CandleSetting.BodyShort)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.close,
        ca.body_high,
        ca.body_low,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.BodyDoji, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
        penetration,
    )


def cdl_eveningdojistar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: float | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
    # TA-Lib rejects a negative penetration (TA_BAD_PARAM)
    penetration = _number(penetration, 0.3, "penetration", ge=0)
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
