# -*- coding: utf-8 -*-
# Median (MEDIAN)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def median(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Median"""
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

    # Calculate Result
    median = close.rolling(length, min_periods=min_periods).median()

    # Offset
    median = apply_offset(median, offset, **kwargs)

    # Name & Category
    median.name = f"MEDIAN_{length}"
    median.category = "statistics"

    return median


median.__doc__ = """Rolling Median

Rolling Median of over 'n' periods. Sibling of a Simple Moving Average.

Sources:
    https://www.incrediblecharts.com/indicators/median_price.php

Calculation:
    Default Inputs:
        length=30
    MEDIAN = close.rolling(length).median()

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
