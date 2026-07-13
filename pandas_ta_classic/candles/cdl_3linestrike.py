# Candle Three-Line Strike (CDL_3LINESTRIKE)
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
    color,
    O_,
    C,
    arr_nr,
    out,
    start_idx,
    near_trail,
    near_total_3,
    near_total_2,
    f_near,
):
    for i in range(start_idx, len(out)):
        if (
            # Three candles with same color
            color[i - 3] == color[i - 2]
            and color[i - 2] == color[i - 1]
            # 4th opposite color
            and color[i] == -color[i - 1]
            # 2nd opens within/near 1st real body
            and O_[i - 2] >= min(O_[i - 3], C[i - 3]) - f_near * near_total_3
            and O_[i - 2] <= max(O_[i - 3], C[i - 3]) + f_near * near_total_3
            # 3rd opens within/near 2nd real body
            and O_[i - 1] >= min(O_[i - 2], C[i - 2]) - f_near * near_total_2
            and O_[i - 1] <= max(O_[i - 2], C[i - 2]) + f_near * near_total_2
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

        # Update totals: add current range, subtract trailing range
        near_total_3 += arr_nr[i - 3] - arr_nr[near_trail - 3]
        near_total_2 += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(Near) + 3
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 3
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]

    near_trail = start_idx - near_period

    # Seed Near totals for i-3 and i-2 (indices 3, 2)
    near_total_3 = float(arr_nr[near_trail - 3 : start_idx - 3].sum())
    near_total_2 = float(arr_nr[near_trail - 2 : start_idx - 2].sum())

    _detect_nb(
        ca.color,
        ca.open,
        ca.close,
        arr_nr,
        out,
        start_idx,
        near_trail,
        near_total_3,
        near_total_2,
        AVG_FACTOR[CandleSetting.Near],
    )


def cdl_3linestrike(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
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
