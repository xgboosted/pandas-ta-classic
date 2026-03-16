# -*- coding: utf-8 -*-
# Z Score (ZSCORE)
from typing import Any, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series
from pandas_ta_classic.utils import get_offset, verify_series


def zscore(
    close: Series,
    length: Optional[int] = None,
    std: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Z Score"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    std = float(std) if std and std > 1 else 1
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy for cross-version determinism.
    values = close.values.astype(np.float64)
    n = len(values)
    result_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= length:
        windows = sliding_window_view(values, length)
        window_mean = windows.mean(axis=1)
        window_std = windows.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result_arr[length - 1 :] = (values[length - 1 :] - window_mean) / (
                std * window_std
            )
    zscore = Series(result_arr, index=close.index, dtype=np.float64)

    # Offset
    if offset != 0:
        zscore = zscore.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        zscore.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                zscore.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                zscore.bfill(inplace=True)

    # Name & Category
    zscore.name = f"ZS_{length}"
    zscore.category = "statistics"

    return zscore


zscore.__doc__ = """Rolling Z Score

Sources:

Calculation:
    Default Inputs:
        length=30, std=1
    SMA = Simple Moving Average
    STDEV = Standard Deviation
    std = std * STDEV(close, length)
    mean = SMA(close, length)
    ZSCORE = (close - mean) / std

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    std (float): It's period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
