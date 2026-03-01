# -*- coding: utf-8 -*-
# SuperTrend (SUPERTREND)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


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
    matr = multiplier * atr(high, low, close, length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    from pandas_ta_classic.utils._numba import _supertrend_loop

    c_arr = close.to_numpy()
    ub_arr = upperband.to_numpy().copy()
    lb_arr = lowerband.to_numpy().copy()

    dir_arr, trend_arr, long_arr, short_arr = _supertrend_loop(c_arr, ub_arr, lb_arr, m)

    # Wrap numpy arrays as Series
    _idx = close.index
    trend = Series(trend_arr, index=_idx)
    dir_ = Series(dir_arr, index=_idx)
    long = Series(long_arr, index=_idx)
    short = Series(short_arr, index=_idx)

    # Offset, Name and Categorize it
    _props = f"_{length}_{multiplier}"
    return _build_dataframe(
        {
            f"SUPERT{_props}": trend,
            f"SUPERTd{_props}": dir_,
            f"SUPERTl{_props}": long,
            f"SUPERTs{_props}": short,
        },
        f"SUPERT{_props}",
        "overlap",
        offset,
        **kwargs,
    )


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
