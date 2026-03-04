# -*- coding: utf-8 -*-
# Z Score (ZSCORE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.sma import sma
from .stdev import stdev
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


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

    # Calculate Result
    std *= stdev(close=close, length=length, **kwargs)
    mean = sma(close=close, length=length, **kwargs)
    zscore = (close - mean) / std

    return _finalize(zscore, offset, f"ZS_{length}", "statistics", **kwargs)


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
