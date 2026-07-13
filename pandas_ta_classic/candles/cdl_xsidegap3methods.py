# Candle Upside/Downside Gap Three Methods (CDL_XSIDEGAP3METHODS)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.candles._cdl_math import (
    run_pattern,
)
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(color, open_, close, body_hi, body_lo, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 2] == color[i - 1]
            and color[i - 1] == -color[i]
            and open_[i] < body_hi[i - 1]
            and open_[i] > body_lo[i - 1]
            and close[i] < body_hi[i - 2]
            and close[i] > body_lo[i - 2]
            and ((color[i - 2] == 1 and body_lo[i - 1] > body_hi[i - 2]) or (color[i - 2] == -1 and body_hi[i - 1] < body_lo[i - 2]))
        ):
            out[i] = color[i - 2] * 100


def _detect(ca, out, **kwargs):
    start_idx = 2
    if start_idx >= len(out):
        return

    body_hi = ca.body_high
    body_lo = ca.body_low

    _detect_nb(ca.color, ca.open, ca.close, body_hi, body_lo, out, start_idx)


def cdl_xsidegap3methods(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Upside/Downside Gap Three Methods"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_XSIDEGAP3METHODS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
