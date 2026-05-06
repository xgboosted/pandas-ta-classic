# -*- coding: utf-8 -*-
# Pearson Correlation Coefficient (CORREL)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def correl(
    close: Series,
    benchmark: Optional[Series] = None,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Pearson Correlation Coefficient"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    benchmark = verify_series(benchmark, max(length, min_periods))
    offset = get_offset(offset)

    if close is None or benchmark is None:
        return None

    # Calculate Result
    result = close.rolling(length, min_periods=min_periods).corr(benchmark)

    # Offset
    result = apply_offset(result, offset)

    result = apply_fill(result, **kwargs)

    # Name and Categorize it
    result.name = f"CORREL_{length}"
    result.category = "statistics"

    return result


correl.__doc__ = """Pearson Correlation Coefficient (CORREL)

The Pearson Correlation Coefficient measures the linear relationship between
two series over a rolling window.  Values range from -1 (perfect negative
correlation) to +1 (perfect positive correlation).

Sources:
    https://www.investopedia.com/terms/c/correlationcoefficient.asp

Calculation:
    Default Inputs:
        length=30
    CORREL = close.rolling(length).corr(benchmark)

Args:
    close (pd.Series): Series of 'close's
    benchmark (pd.Series): Series of benchmark 'close's
    length (int): The period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    min_periods (int): Minimum observations required. Default: length
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
