# -*- coding: utf-8 -*-
# Linear Regression Angle (LINEARREG_ANGLE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series


def linregangle(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Linear Regression Angle (LINEARREG_ANGLE)

    The angle (in degrees) of the linear regression slope.
    TA-Lib name: LINEARREG_ANGLE.
    """
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    result = linreg(close, length=length, offset=offset, angle=True, degrees=True)
    return result


linregangle.__doc__ = """Linear Regression Angle (LINEARREG_ANGLE)

Returns the angle (in degrees) of the linear regression line over the
last *length* bars.  Equivalent to ta.linreg(..., angle=True).

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 14
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
