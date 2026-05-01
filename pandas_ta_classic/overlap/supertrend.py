# -*- coding: utf-8 -*-
# SuperTrend (SUPERTREND)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _supertrend_loop(c_arr, ub_arr, lb_arr, m):
    dir_ = np.ones(m)
    trend = np.zeros(m)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)
    for i in range(1, m):
        if c_arr[i] > ub_arr[i - 1]:
            dir_[i] = 1.0
        elif c_arr[i] < lb_arr[i - 1]:
            dir_[i] = -1.0
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lb_arr[i] < lb_arr[i - 1]:
                lb_arr[i] = lb_arr[i - 1]
            if dir_[i] < 0 and ub_arr[i] > ub_arr[i - 1]:
                ub_arr[i] = ub_arr[i - 1]
        if dir_[i] > 0:
            trend[i] = lb_arr[i]
            long[i] = lb_arr[i]
        else:
            trend[i] = ub_arr[i]
            short[i] = ub_arr[i]
    return dir_, trend, long, short


def supertrend(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    multiplier: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Supertrend"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 7
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Results
    m = close.size

    hl2_ = hl2(high, low)
    _atr = atr(high, low, close, length)
    if _atr is None:
        return None
    matr = multiplier * _atr
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    c_arr = close.to_numpy(dtype=float)
    ub_arr = upperband.to_numpy(dtype=float, copy=True)
    lb_arr = lowerband.to_numpy(dtype=float, copy=True)
    dir_, trend, long, short = _supertrend_loop(c_arr, ub_arr, lb_arr, m)

    # Prepare DataFrame to return
    _props = f"_{length}_{multiplier}"
    df = DataFrame(
        {
            f"SUPERT{_props}": trend,
            f"SUPERTd{_props}": dir_,
            f"SUPERTl{_props}": long,
            f"SUPERTs{_props}": short,
        },
        index=close.index,
    )

    df.name = f"SUPERT{_props}"
    df.category = "overlap"

    # Offset
    df = apply_offset(df, offset)

    # Handle fills
    df = apply_fill(df, **kwargs)

    return df


supertrend.__doc__ = """Supertrend (supertrend)

Supertrend is an overlap indicator. It is used to help identify trend
direction, setting stop loss, identify support and resistance, and/or
generate buy & sell signals.

Sources:
    http://www.freebsensetips.com/blog/detail/7/What-is-supertrend-indicator-its-calculation

Calculation:
    Default Inputs:
        length=7, multiplier=3.0
    Default Direction:
	Set to +1 or bullish trend at start

    MID = multiplier * ATR
    LOWERBAND = HL2 - MID
    UPPERBAND = HL2 + MID

    if UPPERBAND[i] < FINAL_UPPERBAND[i-1] and close[i-1] > FINAL_UPPERBAND[i-1]:
        FINAL_UPPERBAND[i] = UPPERBAND[i]
    else:
        FINAL_UPPERBAND[i] = FINAL_UPPERBAND[i-1])

    if LOWERBAND[i] > FINAL_LOWERBAND[i-1] and close[i-1] < FINAL_LOWERBAND[i-1]:
        FINAL_LOWERBAND[i] = LOWERBAND[i]
    else:
        FINAL_LOWERBAND[i] = FINAL_LOWERBAND[i-1])

    if close[i] <= FINAL_UPPERBAND[i]:
        SUPERTREND[i] = FINAL_UPPERBAND[i]
    else:
        SUPERTREND[i] = FINAL_LOWERBAND[i]

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int) : length for ATR calculation. Default: 7
    multiplier (float): Coefficient for upper and lower band distance to
        midrange. Default: 3.0
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: SUPERT (trend), SUPERTd (direction), SUPERTl (long), SUPERTs (short) columns.
"""
