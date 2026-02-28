# -*- coding: utf-8 -*-
# Quantile (QUANTILE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def quantile(
    close: Series,
    length: Optional[int] = None,
    q: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Quantile"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    q = float(q) if q and q > 0 and q < 1 else 0.5
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    quantile = close.rolling(length, min_periods=min_periods).quantile(q)

    # Offset
    quantile = apply_offset(quantile, offset, **kwargs)

    # Name & Category
    quantile.name = f"QTL_{length}_{q}"
    quantile.category = "statistics"

    return quantile


quantile.__doc__ = """Rolling Quantile

Sources:

Calculation:
    Default Inputs:
        length=30, q=0.5
    QUANTILE = close.rolling(length).quantile(q)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    q (float): The quantile. Default: 0.5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
