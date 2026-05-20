# Variance (VARIANCE)
from typing import Any, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def _numpy_rolling_variance(values, length, ddof, min_periods):
    """Compute rolling variance using NumPy, respecting *min_periods*.

    For windows of size *length* and beyond the standard
    :func:`numpy.lib.stride_tricks.sliding_window_view` approach is used.
    When *min_periods* is smaller than *length* the leading partial windows
    are filled individually so that they are not left as ``NaN``.

    Args:
        values (np.ndarray): 1-D float64 price array.
        length (int): Rolling window size.
        ddof (int): Delta degrees of freedom passed to ``ndarray.var``.
        min_periods (int): Minimum number of observations required to produce
            a non-NaN result.

    Returns:
        np.ndarray: Rolling variance array of the same length as *values*.
    """
    n = len(values)
    result_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= length:
        windows = sliding_window_view(values, length)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            result_arr[length - 1 :] = windows.var(axis=1, ddof=ddof)
    if min_periods < length:
        for pos in range(min_periods - 1, min(length - 1, n)):
            w = values[: pos + 1]
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                result_arr[pos] = w.var(ddof=ddof) if len(w) > ddof else np.nan
    return result_arr


def variance(
    close: Series,
    length: Optional[int] = None,
    ddof: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Variance"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 0
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import VAR

        variance = VAR(close, length)
    else:
        result_arr = _numpy_rolling_variance(
            close.values.astype(np.float64), length, ddof, min_periods
        )
        variance = Series(result_arr, index=close.index, dtype=np.float64)

    # Offset
    variance = apply_offset(variance, offset)

    variance = apply_fill(variance, **kwargs)

    # Name & Category
    variance.name = f"VAR_{length}"
    variance.category = "statistics"

    return variance


variance.__doc__ = """Rolling Variance

Sources:

Calculation:
    Default Inputs:
        length=30
    VARIANCE = close.rolling(length).var()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 0
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
