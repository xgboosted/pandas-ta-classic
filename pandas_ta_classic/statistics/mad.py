# -*- coding: utf-8 -*-
# Mean Absolute Deviation (MAD)
from typing import Any, Optional
import numpy as np
from numpy import fabs as npfabs
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def mad(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Mean Absolute Deviation"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result — fully vectorised via sliding_window_view (avoids
    # ~5 200 Python callbacks from rolling.apply).
    c_arr = close.to_numpy(dtype=float)
    windows = sliding_window_view(c_arr, length)  # (n-L+1, L)
    means = windows.mean(axis=1)
    mad_vals = np.abs(windows - means[:, np.newaxis]).mean(axis=1)
    result = np.full(len(c_arr), np.nan)
    result[length - 1 :] = mad_vals
    mad = Series(result, index=close.index)

    # Offset
    mad = apply_offset(mad, offset, **kwargs)

    # Name & Category
    mad.name = f"MAD_{length}"
    mad.category = "statistics"

    return mad


mad.__doc__ = """Rolling Mean Absolute Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    mad = close.rolling(length).mad()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
