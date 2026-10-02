# Candle Three-Line Strike (CDL_3LINESTRIKE)
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
def _detect_nb(color, O_, C, near_3, near_2, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # Three candles with same color
            color[i - 3] == color[i - 2]
            and color[i - 2] == color[i - 1]
            # 4th opposite color
            and color[i] == -color[i - 1]
            # 2nd opens within/near 1st real body
            and O_[i - 2] >= min(O_[i - 3], C[i - 3]) - near_3[i]
            and O_[i - 2] <= max(O_[i - 3], C[i - 3]) + near_3[i]
            # 3rd opens within/near 2nd real body
            and O_[i - 1] >= min(O_[i - 2], C[i - 2]) - near_2[i]
            and O_[i - 1] <= max(O_[i - 2], C[i - 2]) + near_2[i]
            and (
                (
                    # If three white
                    color[i - 1] == 1
                    # Consecutive higher closes
                    and C[i - 1] > C[i - 2]
                    and C[i - 2] > C[i - 3]
                    # 4th opens above prior close
                    and O_[i] > C[i - 1]
                    # 4th closes below 1st open
                    and C[i] < O_[i - 3]
                )
                or (
                    # If three black
                    color[i - 1] == -1
                    # Consecutive lower closes
                    and C[i - 1] < C[i - 2]
                    and C[i - 2] < C[i - 3]
                    # 4th opens below prior close
                    and O_[i] < C[i - 1]
                    # 4th closes above 1st open
                    and C[i] > O_[i - 3]
                )
            )
        ):
            out[i] = color[i - 1] * 100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(Near) + 3
    start_idx = candle_avg_period(CandleSetting.Near) + 3
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.Near, 3, start_idx),
        candle_average(ca, CandleSetting.Near, 2, start_idx),
        out,
        start_idx,
    )


def cdl_3linestrike(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Three-Line Strike

    Three same-colored candles with consecutively higher (white) or lower
    (black) closes, each opening within or near the prior real body. The
    fourth candle is the opposite color and engulfs the entire 3-candle
    move (opening beyond the third's close, closing beyond the first's
    open).

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (bullish) / -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_3linestrike(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3LINESTRIKE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
