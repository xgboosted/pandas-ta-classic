# -*- coding: utf-8 -*-
# Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import get_offset, verify_series


def ht_dcperiod(
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Dominant Cycle Period"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ht = hilbert_result(close)
    result = Series(ht["smooth_period"], index=close.index)

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            result.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            result.bfill(inplace=True)

    # Name and Categorize it
    result.name = "HT_DCPERIOD"
    result.category = "cycles"

    return result


ht_dcperiod.__doc__ = """Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD)

The Dominant Cycle Period uses the Hilbert Transform to estimate the
dominant cycle period of the price data.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
