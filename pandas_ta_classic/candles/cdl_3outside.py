# Candle Three Outside Up/Down (CDL_3OUTSIDE)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(color, close, open_, out, start_idx):
    for i in range(start_idx, len(out)):
        if (color[i - 1] == 1 and color[i - 2] == -1 and close[i - 1] > open_[i - 2] and open_[i - 1] < close[i - 2] and close[i] > close[i - 1]) or (
            color[i - 1] == -1 and color[i - 2] == 1 and open_[i - 1] > close[i - 2] and close[i - 1] < open_[i - 2] and close[i] < close[i - 1]
        ):
            out[i] = color[i - 1] * 100


def _detect(ca, out, **kwargs):
    start_idx = 3
    if start_idx >= len(out):
        return

    _detect_nb(ca.color, ca.close, ca.open, out, start_idx)


def cdl_3outside(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Three Outside Up/Down"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3OUTSIDE",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
