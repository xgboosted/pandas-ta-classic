# -*- coding: utf-8 -*-
# Linear Regression (LINREG)
from typing import Any, Optional
import numpy as np
from numpy import array as npArray
from numpy import arctan as npAtan
from numpy import pi as npPi
from pandas import Series

npNaN = np.nan
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def linreg(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Linear Regression"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)
    angle = kwargs.pop("angle", False)
    intercept = kwargs.pop("intercept", False)
    degrees = kwargs.pop("degrees", True)
    r = kwargs.pop("r", False)
    slope = kwargs.pop("slope", False)
    tsf = kwargs.pop("tsf", False)

    if close is None:
        return None

    # Calculate Result — fully vectorised OLS over all windows at once.
    # x = [0, 1, ..., L-1] matches TA-Lib convention; precompute scalar sums.
    x_arr = np.arange(length, dtype=float)
    x_sum = 0.5 * length * (length - 1)
    x2_sum = length * (length - 1) * (2 * length - 1) / 6
    divisor = length * x2_sum - x_sum * x_sum

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(npArray(close, dtype=float), length)  # (n-L+1, L)
    y_sums = windows.sum(axis=1)
    xy_sums = (windows * x_arr).sum(axis=1)
    m_slopes = (length * xy_sums - x_sum * y_sums) / divisor

    if slope:
        linreg_ = m_slopes
    else:
        bs = (y_sums * x2_sum - x_sum * xy_sums) / divisor
        if intercept:
            linreg_ = bs
        elif angle:
            theta = npAtan(m_slopes)
            if degrees:
                theta *= 180 / npPi
            linreg_ = theta
        elif r:
            y2_sums = (windows * windows).sum(axis=1)
            rn = length * xy_sums - x_sum * y_sums
            rd = (divisor * (length * y2_sums - y_sums**2)) ** 0.5
            linreg_ = rn / rd
        elif tsf:
            linreg_ = m_slopes * length + bs
        else:
            linreg_ = m_slopes * (length - 1) + bs

    linreg = Series(
        np.concatenate([[npNaN] * (length - 1), linreg_]), index=close.index
    )

    # Offset
    linreg = apply_offset(linreg, offset)

    linreg = apply_fill(linreg, **kwargs)

    # Name and Categorize it
    name = "LR"
    if slope:
        name += "m"
    if intercept:
        name += "b"
    if angle:
        name += "a"
    if r:
        name += "r"
    name += f"_{length}"
    linreg.name = name
    linreg.category = "overlap"

    return linreg


linreg.__doc__ = """Linear Regression Moving Average (linreg)

Linear Regression Moving Average (LINREG). This is a simplified version of a
Standard Linear Regression. LINREG is a rolling regression of one variable. A
Standard Linear Regression is between two or more variables.

Source: TA Lib

Calculation:
    Default Inputs:
        length=14
    x = [0, 1, ..., n-1]  (matches TA-Lib convention)
    x_sum = 0.5 * length * (length - 1)
    x2_sum = length * (length - 1) * (2 * length - 1) / 6
    divisor = length * x2_sum - x_sum * x_sum

    lr(series):
        y_sum = series.sum()
        y2_sum = (series* series).sum()
        xy_sum = (x * series).sum()

        m = (length * xy_sum - x_sum * y_sum) / divisor
        b = (y_sum * x2_sum - x_sum * xy_sum) / divisor
        return m * (length - 1) + b

    linreg = close.rolling(length).apply(lr)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    angle (bool, optional): If True, returns the angle of the slope.
        Default: False.
    degrees (bool, optional): If True, returns the angle in degrees;
        if False, in radians. Default: True (matches TA-Lib convention).
    intercept (bool, optional): If True, returns the angle of the slope in
        radians. Default: False.
    r (bool, optional): If True, returns it's correlation 'r'. Default: False.
    slope (bool, optional): If True, returns the slope. Default: False.
    tsf (bool, optional): If True, returns the Time Series Forecast value.
        Default: False.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
