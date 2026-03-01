# -*- coding: utf-8 -*-
# Bias (BIAS)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def bias(
    close: Series,
    length: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Bias (BIAS)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 26
    mamode = mamode if isinstance(mamode, str) else "sma"
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    bma = ma(mamode, close, length=length, **kwargs)
    if bma is None:
        return None
    bias = (close / bma) - 1

    return _finalize(bias, offset, f"BIAS_{bma.name}", "momentum", **kwargs)


bias.__doc__ = """Bias (BIAS)

Rate of change between the source and a moving average.

Sources:
    Few internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=26, MA='sma'

    BIAS = (close - MA(close, length)) / MA(close, length)
         = (close / MA(close, length)) - 1

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 26
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    drift (int): The short period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
