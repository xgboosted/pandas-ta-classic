# Candle Engulfing Pattern (CDL_ENGULFING)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    CandleArrays,
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit
import numpy as np


@njit(cache=True)
def _detect_nb(color, open_, close, out, start_idx):
    for i in range(start_idx, len(out)):
        bullish = (
            color[i] == 1
            and color[i - 1] == -1
            and ((close[i] >= open_[i - 1] and open_[i] < close[i - 1]) or (close[i] > open_[i - 1] and open_[i] <= close[i - 1]))
        )
        bearish = (
            color[i] == -1
            and color[i - 1] == 1
            and ((open_[i] >= close[i - 1] and close[i] < open_[i - 1]) or (open_[i] > close[i - 1] and close[i] <= open_[i - 1]))
        )
        if bullish or bearish:
            if open_[i] != close[i - 1] and close[i] != open_[i - 1]:
                out[i] = color[i] * 100
            else:
                out[i] = color[i] * 80


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: 2 (no candle settings needed)
    start_idx = 2
    if start_idx >= len(out):
        return

    _detect_nb(ca.color, ca.open, ca.close, out, start_idx)


def cdl_engulfing(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Engulfing"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ENGULFING",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
