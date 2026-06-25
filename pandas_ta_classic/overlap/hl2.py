# HL2 (HL2)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def hl2(
    high: Series,
    low: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: HL2"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import MEDPRICE

        hl2 = Series(MEDPRICE(high, low), index=high.index)
    else:
        hl2 = 0.5 * (high + low)

    # Offset
    hl2 = apply_offset(hl2, offset)
    hl2 = apply_fill(hl2, **kwargs)

    # Name & Category
    hl2.name = "HL2"
    hl2.category = "overlap"

    return hl2

hl2.__doc__ = """HL2 (Median Price)

HL2 calculates the median price, which is the average of the High and Low 
prices for each period. This indicator is commonly used to represent the 
mid-point of a period's price range and is often used in technical analysis 
as a reference point for price action.

Sources:
    https://www.tradingview.com/support/solutions/43000502274-hl2/
    https://www.investopedia.com/terms/m/median.asp
    https://school.stockcharts.com/doku.php?id=chart_analysis:typical_price

Calculation:
    Default Inputs:
        None (uses raw high and low prices)
    
    HL2 = (High + Low) / 2

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
