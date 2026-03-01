# -*- coding: utf-8 -*-
# Price Volume (PVOL)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import (
    _finalize,
    apply_offset,
    get_offset,
    signed_series,
    verify_series,
)


def pvol(
    close: Series,
    volume: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Price-Volume (PVOL)"""
    # Validate arguments
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)
    signed = kwargs.pop("signed", False)

    if close is None or volume is None:
        return None

    # Calculate Result
    pvol = close * volume
    if signed:
        pvol *= signed_series(close, 1)

    return _finalize(pvol, offset, f"PVOL", "volume", **kwargs)


pvol.__doc__ = """Price-Volume (PVOL)

Returns a series of the product of price and volume.

Calculation:
    if signed:
        pvol = signed_series(close, 1) * close * volume
    else:
        pvol = close * volume

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    signed (bool): Keeps the sign of the difference in 'close's. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
