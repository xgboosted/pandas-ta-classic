# Candle Three Black Crows (CDL_3BLACKCROWS)
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
def _detect_nb(color, lower_shadow, O_, H, C, svs_2, svs_1, svs_0, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            # Prior candle (i-3) is white
            color[i - 3] == 1
            # 1st black
            and color[i - 2] == -1
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
            # 2nd black opens within 1st black's real body
            and O_[i - 1] < O_[i - 2]
            and O_[i - 1] > C[i - 2]
            # 3rd black opens within 2nd black's real body
            and O_[i] < O_[i - 1]
            and O_[i] > C[i - 1]
            # 1st black closes under prior candle's high
            and H[i - 3] > C[i - 2]
            # Three declining closes
            and C[i - 2] > C[i - 1]
            and C[i - 1] > C[i]
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(ShadowVeryShort) + 3
    start_idx = candle_avg_period(CandleSetting.ShadowVeryShort) + 3
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.lower_shadow,
        ca.open,
        ca.high,
        ca.close,
        candle_average(ca, CandleSetting.ShadowVeryShort, 2, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 1, start_idx),
        candle_average(ca, CandleSetting.ShadowVeryShort, 0, start_idx),
        out,
        start_idx,
    )


def cdl_3blackcrows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Three Black Crows

    Three consecutive declining black (bearish) candlesticks, each with
    very short lower shadows. Each candle after the first opens within
    the prior candle's real body. The first candle's close is below the
    preceding white candle's high.

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
        >>> result = cdl_3blackcrows(df.open, df.high, df.low, df.close)
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3BLACKCROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
