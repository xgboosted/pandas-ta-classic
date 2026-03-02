# -*- coding: utf-8 -*-
# Projection Oscillator (PO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import (
    _finalize,
    get_offset,
    non_zero_range,
    verify_series,
)


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

    # Projection oscillator as percentage
    # Use non_zero_range to avoid division by zero
    lr = non_zero_range(lr, lr)
    po = 100 * (close - lr) / lr

    return _finalize(po, offset, f"PO_{length}", "momentum", **kwargs)


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
