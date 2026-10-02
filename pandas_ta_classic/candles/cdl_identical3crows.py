# Candle Identical Three Crows (CDL_IDENTICAL3CROWS)
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
def _detect_nb(color, lower_shadow, O_, C, svs_2, svs_1, svs_0, equal_2, equal_1, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # 1st black
            color[i - 2] == -1
            # very short lower shadow
            and lower_shadow[i - 2] < svs_2[i]
            # 2nd black
            and color[i - 1] == -1
            # very short lower shadow
            and lower_shadow[i - 1] < svs_1[i]
            # 3rd black
            and color[i] == -1
            # very short lower shadow
            and lower_shadow[i] < svs_0[i]
            # Three declining closes
            and C[i - 2] > C[i - 1]
            and C[i - 1] > C[i]
            # 2nd opens very close to 1st close
            and O_[i - 1] <= C[i - 2] + equal_2[i]
            and O_[i - 1] >= C[i - 2] - equal_2[i]
            # 3rd opens very close to 2nd close
            and O_[i] <= C[i - 1] + equal_1[i]
            and O_[i] >= C[i - 1] - equal_1[i]
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(TA_CANDLEAVGPERIOD(ShadowVeryShort),
    #               TA_CANDLEAVGPERIOD(Equal)) + 2
    start_idx = max(candle_avg_period(CandleSetting.ShadowVeryShort), candle_avg_period(CandleSetting.Equal)) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.lower_shadow,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.ShadowVeryShort, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        candle_average(ca, CandleSetting.Equal, 2, start_idx),
        candle_average(ca, CandleSetting.Equal, 1, start_idx),
        out,
        start_idx,
    )


def cdl_identical3crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Identical Three Crows

    Three consecutive declining black (bearish) candlesticks, each with
    very short lower shadows. Each candle after the first opens at or
    very close to the prior candle's close (the "identical" open
    distinguishes this from regular Three Black Crows).

    Args:
        open_: Series of 'open' prices.
        high: Series of 'high' prices.
        low: Series of 'low' prices.
        close: Series of 'close' prices.
        scalar: Multiplier for output values. Default: 100.
        offset: Number of periods to shift the result.

    Returns:
        A Series with -100 (bearish) / 0, or None.

    Example:
        >>> result = cdl_identical3crows(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_IDENTICAL3CROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
