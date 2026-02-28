# -*- coding: utf-8 -*-
# Skew (SKEW)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def skew(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Skew"""
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
    skew = close.rolling(length, min_periods=min_periods).skew()

    # Offset
    skew = apply_offset(skew, offset, **kwargs)

    # Name & Category
    skew.name = f"SKEW_{length}"
    skew.category = "statistics"

    return skew


skew.__doc__ = """Rolling Skew

Sources:

Calculation:
    Default Inputs:
        length=30
    SKEW = close.rolling(length).skew()

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
