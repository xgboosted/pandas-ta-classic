# Linear Regression (LINREG)
from typing import Any, Optional
import numpy as np
from numpy import array as npArray
from numpy import arctan as npAtan
from numpy import pi as npPi
from pandas import Series

npNaN = np.nan
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series

# TA-Lib dispatch map: (angle, intercept, slope, tsf) → (module, function)
_TALIB_DISPATCH = {
    (True, False, False, False): ("talib", "LINEARREG_ANGLE"),
    (False, True, False, False): ("talib", "LINEARREG_INTERCEPT"),
    (False, False, True, False): ("talib", "LINEARREG_SLOPE"),
    (False, False, False, True): ("talib", "TSF"),
    (False, False, False, False): ("talib", "LINEARREG"),
}


def _linreg_output(
    windows,
    x_arr,
    x_sum,
    x2_sum,
    divisor,
    length,
    slope,
    intercept,
    angle,
    degrees,
    r,
    tsf,
):
    """Compute the requested linear-regression output over *windows*.

    All mode flags are mutually exclusive; the first truthy flag wins.  When
    no flag is set the default end-point regression value is returned.

    Args:
        windows (np.ndarray): Shape ``(n - length + 1, length)`` sliding view.
        x_arr (np.ndarray): ``[0, 1, …, length-1]`` index array.
        x_sum (float): Precomputed sum of *x_arr*.
        x2_sum (float): Precomputed sum of squares of *x_arr*.
        divisor (float): ``length * x2_sum - x_sum ** 2``.
        length (int): Regression window length.
        slope (bool): Return OLS slope.
        intercept (bool): Return OLS intercept.
        angle (bool): Return slope angle.
        degrees (bool): Angle in degrees (``True``) or radians (``False``).
        r (bool): Return Pearson correlation coefficient.
        tsf (bool): Return Time Series Forecast (one step ahead).

    Returns:
        np.ndarray: 1-D array of regression values for each window.
    """
    y_sums = windows.sum(axis=1)
    xy_sums = (windows * x_arr).sum(axis=1)
    m_slopes = (length * xy_sums - x_sum * y_sums) / divisor

    if slope:
        return m_slopes

    bs = (y_sums * x2_sum - x_sum * xy_sums) / divisor

    if intercept:
        return bs
    if angle:
        theta = npAtan(m_slopes)
        if degrees:
            theta *= 180 / npPi
        return theta
    if r:
        y2_sums = (windows * windows).sum(axis=1)
        rn = length * xy_sums - x_sum * y_sums
        rd = (divisor * (length * y2_sums - y_sums**2)) ** 0.5
        return rn / rd
    if tsf:
        return m_slopes * length + bs
    return m_slopes * (length - 1) + bs


def linreg(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
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
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # TA-Lib dispatch (not available for `r` or angle-in-radians).
    _use_talib = (
        Imports["talib"] and mode_talib and not r and not (angle and not degrees)
    )
    if _use_talib:
        _close_arr = npArray(close, dtype=float)
        _talib_fn = _TALIB_DISPATCH.get((angle, intercept, slope, tsf))
        if _talib_fn is not None:
            from importlib import import_module

            mod_name, fn_name = _talib_fn
            mod = import_module(mod_name)
            _talib_out = getattr(mod, fn_name)(_close_arr, timeperiod=length)
            linreg = Series(np.array(_talib_out, dtype=float), index=close.index)
        else:
            _use_talib = False  # fall through to native

    if not _use_talib:
        # Calculate Result — fully vectorised OLS over all windows at once.
        # x = [0, 1, ..., L-1] matches TA-Lib convention; precompute scalar sums.
        x_arr = np.arange(length, dtype=float)
        x_sum = 0.5 * length * (length - 1)
        x2_sum = length * (length - 1) * (2 * length - 1) / 6
        divisor = length * x2_sum - x_sum * x_sum

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(npArray(close, dtype=float), length)  # (n-L+1, L)
        linreg_ = _linreg_output(
            windows,
            x_arr,
            x_sum,
            x2_sum,
            divisor,
            length,
            slope,
            intercept,
            angle,
            degrees,
            r,
            tsf,
        )

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
