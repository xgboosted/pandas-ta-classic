# -*- coding: utf-8 -*-
# HLC3 (HLC3)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def hlc3(
    high: Series,
    low: Series,
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: HLC3"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import TYPPRICE

        hlc3 = TYPPRICE(high, low, close)
    else:
        hlc3 = (high + low + close) / 3.0

    # Offset
    hlc3 = apply_offset(hlc3, offset)

    # Name & Category
    hlc3.name = "HLC3"
    hlc3.category = "overlap"

    return hlc3


hlc3.__doc__ = """HLC3 (Typical Price)

HLC3 calculates the typical price, which is the average of the High, Low, 
and Close prices for each period. This indicator provides a simple measure 
of the average price level during a period and is widely used in technical 
analysis. It's also known as the TA-Lib TYPPRICE function.

Sources:
    https://www.tradingview.com/support/solutions/43000502273-hlc3/
    https://school.stockcharts.com/doku.php?id=chart_analysis:typical_price
    https://www.investopedia.com/terms/t/typicalprice.asp

Calculation:
    Default Inputs:
        None (uses raw HLC prices)
    
    HLC3 = (High + Low + Close) / 3

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    close (pd.Series): Series of 'close' prices
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
