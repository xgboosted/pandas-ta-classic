# Candle Ladder Bottom (CDL_LADDERBOTTOM)
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
def _detect_nb(color, upper_shadow, O_, H, C, svs_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # First three are black candlesticks
            color[i - 4] == -1
            and color[i - 3] == -1
            and color[i - 2] == -1
            # With consecutively lower opens
            and O_[i - 4] > O_[i - 3]
            and O_[i - 3] > O_[i - 2]
            # And consecutively lower closes
            and C[i - 4] > C[i - 3]
            and C[i - 3] > C[i - 2]
            # 4th: black with an upper shadow
            and color[i - 1] == -1
            and upper_shadow[i - 1] > svs_1[i]
            # 5th: white
            and color[i] == 1
            # That opens above prior candle's body (open, since bearish)
            and O_[i] > O_[i - 1]
            # And closes above prior candle's high
            and C[i] > H[i - 1]
        ):
            out[i] = 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 4
    start_idx = candle_avg_period(CandleSetting.ShadowVeryShort) + 4
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.upper_shadow,
        ca.open,
        ca.high,
        ca.close,
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        out,
        start_idx,
    )


def cdl_ladderbottom(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Ladder Bottom

    A 5-candle bullish reversal pattern. Three consecutive black candles
    with lower opens and closes, followed by a black candle with a
    notable upper shadow, then a white candle that opens above the
    fourth candle's body and closes above its high.

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
        >>> result = cdl_ladderbottom(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_LADDERBOTTOM",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
