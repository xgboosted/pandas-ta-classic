# Candle Two Crows (CDL_2CROWS)
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
def _detect_nb(color, real_body, body_hi, body_lo, open_, close, body_long_2, out, start_idx):
    for i in range(start_idx, len(out)):
        if (
            color[i - 2] == 1  # 1st: white
            and real_body[i - 2] > body_long_2[i]  # long
            and color[i - 1] == -1  # 2nd: black
            and body_lo[i - 1] > body_hi[i - 2]  # gapping up
            and color[i] == -1  # 3rd: black
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]  # opening within 2nd rb
            and close[i] > open_[i - 2]
            and close[i] < close[i - 2]  # closing within 1st rb
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 2
    start_idx = candle_avg_period(CandleSetting.BodyLong) + 2
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.body_high,
        ca.body_low,
        ca.open,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 2, start_idx),
        out,
        start_idx,
    )


def cdl_2crows(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
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
