# SuperTrend (SUPERTREND)
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _supertrend_loop(c_arr, ub_arr, lb_arr, m):
    dir_ = np.full(m, np.nan)
    trend = np.full(m, np.nan)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)
    # Ratchet BOTH bands independently on every bar with the "or" rule from the
    # cited source and TradingView.  The previous loop ratcheted only the active
    # band, so the stop level (SUPERT) sat on the wrong side of price near flips.
    final_ub = np.full(m, np.nan)
    final_lb = np.full(m, np.nan)
    for i in range(1, m):
        if np.isnan(final_ub[i - 1]) or np.isnan(final_lb[i - 1]):
            # Warm-up: the previous final band is not established yet (the ATR
            # is still NaN).  Seed from the raw band so NaN does not propagate
            # through the whole series.
            final_ub[i] = ub_arr[i]
            final_lb[i] = lb_arr[i]
            direction = 1.0
        else:
            if ub_arr[i] < final_ub[i - 1] or c_arr[i - 1] > final_ub[i - 1]:
                final_ub[i] = ub_arr[i]
            else:
                final_ub[i] = final_ub[i - 1]
            if lb_arr[i] > final_lb[i - 1] or c_arr[i - 1] < final_lb[i - 1]:
                final_lb[i] = lb_arr[i]
            else:
                final_lb[i] = final_lb[i - 1]

            if c_arr[i] > final_ub[i - 1]:
                direction = 1.0
            elif c_arr[i] < final_lb[i - 1]:
                direction = -1.0
            else:
                direction = dir_[i - 1]

        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            # The ATR is still NaN, so there is no band and no stop level.  Bar 0
            # and the warm-up bars stay NaN instead of reporting the array's
            # initial value as a level of 0.0 in an uptrend.
            continue

        dir_[i] = direction
        if direction > 0:
            trend[i] = final_lb[i]
            long[i] = final_lb[i]
        else:
            trend[i] = final_ub[i]
            short[i] = final_ub[i]
    return dir_, trend, long, short


@nan_on_short_input
@skip_leading_nan("high", "low", "close")
def supertrend(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    multiplier: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Supertrend"""
    # Validate Arguments
    length = _pos_int(length, 7, "length")
    multiplier = _pos_float(multiplier, 3.0, "multiplier")
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Results
    from pandas_ta_classic.volatility.atr import atr

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
    return apply_fill(df, **kwargs)


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

    if UPPERBAND[i] < FINAL_UPPERBAND[i-1] or close[i-1] > FINAL_UPPERBAND[i-1]:
        FINAL_UPPERBAND[i] = UPPERBAND[i]
    else:
        FINAL_UPPERBAND[i] = FINAL_UPPERBAND[i-1]

    if LOWERBAND[i] > FINAL_LOWERBAND[i-1] or close[i-1] < FINAL_LOWERBAND[i-1]:
        FINAL_LOWERBAND[i] = LOWERBAND[i]
    else:
        FINAL_LOWERBAND[i] = FINAL_LOWERBAND[i-1]

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
        All four are NaN until the ATR exists; there is no direction before the
        first band.
"""
