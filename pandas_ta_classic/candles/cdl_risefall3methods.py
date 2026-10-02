# Candle Rising/Falling Three Methods (CDL_RISEFALL3METHODS)
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
def _detect_nb(real_body, color, O_, H, L, C, body_long_4, body_short_3, body_short_2, body_short_1, body_long_0, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # 1st long, then 3 small, 5th long
            real_body[i - 4] > body_long_4[i]
            and real_body[i - 3] < body_short_3[i]
            and real_body[i - 2] < body_short_2[i]
            and real_body[i - 1] < body_short_1[i]
            and real_body[i] > body_long_0[i]
            # white, 3 black, white  ||  black, 3 white, black
            and color[i - 4] == -color[i - 3]
            and color[i - 3] == color[i - 2]
            and color[i - 2] == color[i - 1]
            and color[i - 1] == -color[i]
            # 2nd to 4th hold within 1st: part of real body within 1st range
            and min(O_[i - 3], C[i - 3]) < H[i - 4]
            and max(O_[i - 3], C[i - 3]) > L[i - 4]
            and min(O_[i - 2], C[i - 2]) < H[i - 4]
            and max(O_[i - 2], C[i - 2]) > L[i - 4]
            and min(O_[i - 1], C[i - 1]) < H[i - 4]
            and max(O_[i - 1], C[i - 1]) > L[i - 4]
            # 2nd to 4th are falling (rising)
            and C[i - 2] * color[i - 4] < C[i - 3] * color[i - 4]
            and C[i - 1] * color[i - 4] < C[i - 2] * color[i - 4]
            # 5th opens above (below) the prior close
            and O_[i] * color[i - 4] > C[i - 1] * color[i - 4]
            # 5th closes above (below) the 1st close
            and C[i] * color[i - 4] > C[i - 4] * color[i - 4]
        ):
            out[i] = 100 * color[i - 4]


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(TA_CANDLEAVGPERIOD(BodyShort), TA_CANDLEAVGPERIOD(BodyLong)) + 4
    start_idx = max(candle_avg_period(CandleSetting.BodyShort), candle_avg_period(CandleSetting.BodyLong)) + 4
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.real_body,
        ca.color,
        ca.open,
        ca.high,
        ca.low,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 4, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 3, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 2, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyShort, 1, start_idx, sequential_seed=True),
        candle_average(ca, CandleSetting.BodyLong, 0, start_idx, sequential_seed=True),
        out,
        start_idx,
    )


def cdl_risefall3methods(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Rising/Falling Three Methods

    A 5-candle continuation pattern. Rising Three Methods: long white
    candle, three small declining black candles held within the first's
    range, then a long white candle closing above the first's close.
    Falling Three Methods is the bearish mirror.

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with +100 (rising) / -100 (falling) / 0, or None.

    Example:
        >>> result = cdl_risefall3methods(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_RISEFALL3METHODS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
