# Candle Two Crows (CDL_2CROWS)
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
    real_body,
    body_hi,
    body_lo,
    open_,
    close,
    arr_bl,
    out,
    start_idx,
    body_long_trail,
    body_long_total,
    f_bl,
):
    for i in range(start_idx, len(out)):
        if (
            color[i - 2] == 1  # 1st: white
            and real_body[i - 2] > f_bl * body_long_total  # long
            and color[i - 1] == -1  # 2nd: black
            and body_lo[i - 1] > body_hi[i - 2]  # gapping up
            and color[i] == -1  # 3rd: black
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]  # opening within 2nd rb
            and close[i] > open_[i - 2]
            and close[i] < close[i - 2]  # closing within 1st rb
        ):
            out[i] = -100

        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_long_trail += 1


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 2
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 2 - body_long_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())

    _detect_nb(
        ca.color,
        ca.real_body,
        body_hi,
        body_lo,
        ca.open,
        ca.close,
        arr_bl,
        out,
        start_idx,
        body_long_trail,
        body_long_total,
        AVG_FACTOR[CandleSetting.BodyLong],
    )


def cdl_2crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Candle Pattern: Two Crows"""
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_2CROWS",
        scalar=scalar,
        offset=offset,
        **kwargs,
    )
