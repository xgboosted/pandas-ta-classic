# Candle Abandoned Baby (CDL_ABANDONEDBABY)
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
def _detect_nb(real_body, color, close, hi, lo, body_long_2, body_doji_1, body_short_0, out, start_idx, penetration):
    for i in range(start_idx, len(out)):
        # Pattern detection
        if (
            # 1st: long real body
            real_body[i - 2] > body_long_2[i]
            # 2nd: doji
            and real_body[i - 1] <= body_doji_1[i]
            # 3rd: longer than short
            and real_body[i] > body_short_0[i]
            and (
                (
                    # Bullish 1st white, bearish 3rd black
                    color[i - 2] == 1
                    and color[i] == -1
                    # 3rd closes well within 1st rb
                    and close[i] < close[i - 2] - real_body[i - 2] * penetration
                    # upside candle gap between 1st and 2nd
                    and lo[i - 1] > hi[i - 2]
                    # downside candle gap between 2nd and 3rd
                    and hi[i] < lo[i - 1]
                )
                or (
                    # Bearish 1st black, bullish 3rd white
                    color[i - 2] == -1
                    and color[i] == 1
                    # 3rd closes well within 1st rb
                    and close[i] > close[i - 2] + real_body[i - 2] * penetration
                    # downside candle gap between 1st and 2nd
                    and hi[i - 1] < lo[i - 2]
                    # upside candle gap between 2nd and 3rd
                    and lo[i] > hi[i - 1]
                )
            )
        ):
            out[i] = color[i] * 100


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
        ca.high,
        ca.low,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        candle_average(ca, CandleSetting.BodyDoji, 1, start_idx),
        candle_average(ca, CandleSetting.BodyShort, 0, start_idx),
        out,
        start_idx,
        penetration,
    )


def cdl_abandonedbaby(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: float | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
    # TA-Lib rejects a negative penetration (TA_BAD_PARAM)
    penetration = _number(penetration, 0.3, "penetration", ge=0)
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
