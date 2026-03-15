# -*- coding: utf-8 -*-
# Mean Absolute Deviation (MAD)
from typing import Any, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series
from pandas_ta_classic.utils import get_offset, verify_series


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

    # Pure numpy for cross-version determinism.
    values = close.values.astype(np.float64)
    n = len(values)
    result_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= length:
        windows = sliding_window_view(values, length)
        means = windows.mean(axis=1, keepdims=True)
        result_arr[length - 1 :] = np.abs(windows - means).mean(axis=1)
    if min_periods < length:
        for pos in range(min_periods - 1, min(length - 1, n)):
            w = values[: pos + 1]
            result_arr[pos] = np.abs(w - w.mean()).mean()
    mad = Series(result_arr, index=close.index, dtype=np.float64)

    # Offset
    if offset != 0:
        mad = mad.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mad.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                mad.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                mad.bfill(inplace=True)

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
