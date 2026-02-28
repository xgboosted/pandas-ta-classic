# -*- coding: utf-8 -*-
# Wilder's Moving Average (RMA)
from typing import Any, Optional
import numpy as np
from pandas import Series

npNaN = np.nan
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

    # Calculate Result — SMA-seeded Wilder smoothing (matches TA-Lib)
    close = close.copy()
    sma_nth = close[0:length].mean()
    close[: length - 1] = npNaN
    close.iloc[length - 1] = sma_nth
    rma = close.ewm(alpha=alpha, adjust=False).mean()

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
    alpha = 1 / length
    SMA_nth = SMA(close, length)
    close[:length - 1] = NaN
    close[length - 1] = SMA_nth
    RMA = EWM(close, alpha=alpha, adjust=False)

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
