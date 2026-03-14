# -*- coding: utf-8 -*-
# Kurtosis (KURTOSIS)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import get_offset, np_rolling_moments, verify_series


def kurtosis(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Kurtosis"""
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

    # Pure numpy rolling excess kurtosis (Fisher) for cross-version determinism.
    m2, m4 = np_rolling_moments(close.values, length, 2, 4)
    nf = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = nf * (nf + 1) * (nf - 1) * m4
        denom = (nf - 2) * (nf - 3) * m2**2
        adj = 3.0 * (nf - 1) ** 2 / ((nf - 2) * (nf - 3))
        result = numer / denom - adj
    kurtosis = Series(result, index=close.index, dtype=np.float64)

    # Offset
    if offset != 0:
        kurtosis = kurtosis.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        kurtosis.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                kurtosis.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                kurtosis.bfill(inplace=True)

    # Name & Category
    kurtosis.name = f"KURT_{length}"
    kurtosis.category = "statistics"

    return kurtosis


kurtosis.__doc__ = """Rolling Kurtosis

Sources:

Calculation:
    Default Inputs:
        length=30
    KURTOSIS = close.rolling(length).kurt()

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
