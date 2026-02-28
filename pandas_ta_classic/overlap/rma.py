# -*- coding: utf-8 -*-
# Wilder's Moving Average (RMA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def rma(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: wildeR's Moving Average (RMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    rma = close.ewm(alpha=alpha, min_periods=length).mean()

    # Offset
    rma = apply_offset(rma, offset, **kwargs)

    # Name & Category
    rma.name = f"RMA_{length}"
    rma.category = "overlap"

    return rma


rma.__doc__ = """Wilder's Moving Average (RMA)

Wilder's Moving Average is simply an Exponential Moving Average (EMA) with
a modified alpha = 1 / length.

Sources:
    https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
    https://www.incrediblecharts.com/indicators/wilder_moving_average.php

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    alpha = 1 / length
    RMA = EMA(close, alpha=alpha)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
