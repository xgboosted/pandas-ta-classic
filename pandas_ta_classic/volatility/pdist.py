# -*- coding: utf-8 -*-
# Price Distance (PDIST)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import (
    _finalize,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def pdist(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Price Distance (PDIST)"""
    # Validate Arguments
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    pdist = 2 * non_zero_range(high, low)
    pdist += non_zero_range(open_, close.shift(drift)).abs()
    pdist -= non_zero_range(close, open_).abs()

    return _finalize(pdist, offset, "PDIST", "volatility", **kwargs)


pdist.__doc__ = """Price Distance (PDIST)

Measures the "distance" covered by price movements.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/pricedistance/

Calculation:
    Default Inputs:
        drift=1

    PDIST = 2(high - low) - ABS(close - open) + ABS(open - close[drift])

Args:
    open_ (pd.Series): Series of 'opens's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
