# Candle Dark Cloud Cover (CDL_DARKCLOUDCOVER)
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
from pandas_ta_classic.utils._core import _number
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _detect_nb(color, real_body, open_, high, close, body_long_1, out, start_idx, penetration):
    for i in range(start_idx, len(out)):
        if (
            color[i - 1] == 1  # 1st: white
            and real_body[i - 1] > body_long_1[i]  # long
            and color[i] == -1  # 2nd: black
            and open_[i] > high[i - 1]  # open above prior high
            and close[i] > open_[i - 1]  # close within prior body
            and close[i] < close[i - 1] - real_body[i - 1] * penetration
        ):
            out[i] = -100


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    penetration = kwargs["penetration"]

    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 1
    start_idx = candle_avg_period(CandleSetting.BodyLong) + 1
    if start_idx >= len(out):
        return

    _detect_nb(
        ca.color,
        ca.real_body,
        ca.open,
        ca.high,
        ca.close,
        candle_average(ca, CandleSetting.BodyLong, 1, start_idx, sequential_seed=True),
        out,
        start_idx,
        penetration,
    )


def cdl_darkcloudcover(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    penetration: float | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Candle Pattern: Dark Cloud Cover"""
    # TA-Lib rejects a negative penetration (TA_BAD_PARAM)
    penetration = _number(penetration, 0.5, "penetration", ge=0)
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DARKCLOUDCOVER",
        scalar=scalar,
        offset=offset,
        penetration=penetration,
        **kwargs,
    )
