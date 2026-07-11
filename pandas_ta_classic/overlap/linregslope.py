# Linear Regression Slope (LINEARREG_SLOPE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series


def linregslope(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Linear Regression Slope (LINEARREG_SLOPE)

    The slope of the linear regression line.
    TA-Lib name: LINEARREG_SLOPE.
    """
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    return linreg(close, length=length, talib=talib, offset=offset, slope=True)


linregslope.__doc__ = """Linear Regression Slope (LINEARREG_SLOPE)

Returns the slope of the linear regression line over the last *length*
bars.  Equivalent to ta.linreg(..., slope=True).

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 14
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
