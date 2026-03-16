# -*- coding: utf-8 -*-
# Projection Oscillator (PO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series


def po(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Projection Oscillator (PO)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    # Linear regression
    lr = linreg(close, length=length)
    if lr is None:
        return None

    # Projection oscillator as percentage (protect against lr=0)
    po = 100 * (close - lr) / lr.replace(0, float("nan"))

    # Offset
    if offset != 0:
        po = po.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        po.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            po.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            po.bfill(inplace=True)

    # Name and Categorize it
    po.name = f"PO_{length}"
    po.category = "momentum"

    return po


po.__doc__ = """Projection Oscillator (PO)

The Projection Oscillator measures the percentage deviation of price from its
linear regression trend line. It helps identify overbought and oversold conditions
relative to the trend.

Sources:
    https://www.tradingview.com/script/CDdh2vTz-Projection-Oscillator/
    Technical Analysis of Stock Trends by Edwards & Magee

Calculation:
    Default Inputs:
        length=14

    LR = Linear Regression(close, length)
    PO = 100 * (close - LR) / LR

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
