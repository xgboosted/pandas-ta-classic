# -*- coding: utf-8 -*-
# Accumulation/Distribution (AD)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _finalize,
    apply_offset,
    get_offset,
    non_zero_range,
    verify_series,
)


def ad(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    open_: Optional[Series] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Accumulation/Distribution (AD)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import AD

        ad = AD(high, low, close, volume)
    else:
        if open_ is not None:
            open_ = verify_series(open_)
            ad = non_zero_range(close, open_)  # AD with Open
        else:
            ad = 2 * close - (high + low)  # AD with High, Low, Close

        high_low_range = non_zero_range(high, low)
        ad *= volume / high_low_range
        ad = ad.cumsum()

    return _finalize(ad, offset, "AD" if open_ is None else "ADo", "volume", **kwargs)


ad.__doc__ = """Accumulation/Distribution (AD)

Accumulation/Distribution indicator utilizes the relative position
of the close to it's High-Low range with volume.  Then it is cumulated.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/accumulationdistribution-ad/

Calculation:
    CUM = Cumulative Sum
    if 'open':
        AD = close - open
    else:
        AD = 2 * close - high - low

    hl_range = high - low
    AD = AD * volume / hl_range
    AD = CUM(AD)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    open (pd.Series): Series of 'open's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
