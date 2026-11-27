# Normalized Average True Range (NATR)
from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import Bound, Flag, OneOf, _positive_inputs, nan_on_short_input

from .atr import atr


@nan_on_short_input
@_positive_inputs("close")
def natr(
    high: Series,
    low: Series,
    close: Series,
    length: Annotated[int, Bound(gt=0)] = 14,
    scalar: Annotated[float, Bound()] = 100,
    mamode: Annotated[str, OneOf(tuple(ma()))] = 'rma',
    talib: Annotated[bool, Flag()] = False,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Normalized Average True Range (NATR)"""
    # Validate arguments
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = talib

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib cannot express a non-default drift, mamode, scalar; run natively instead of ignoring it
    if Imports["talib"] and mode_talib and scalar == 100 and mamode == "rma" and drift == 1:
        from talib import NATR

        natr = NATR(high, low, close, length)
    else:
        natr = scalar / close
        _atr = atr(
            high=high,
            low=low,
            close=close,
            length=length,
            mamode=mamode,
            talib=False,
            drift=drift,
        )
        if _atr is None:
            return None
        natr *= _atr

    # Offset
    natr = apply_offset(natr, offset)

    natr = apply_fill(natr, **kwargs)

    # Name and Categorize it
    natr.name = f"NATR_{length}"
    natr.category = "volatility"

    return natr


natr.__doc__ = """Normalized Average True Range (NATR)

Normalized Average True Range attempt to normalize the average true range.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

Calculation:
    Default Inputs:
        length=20
    ATR = Average True Range
    NATR = (100 / close) * ATR(high, low, close)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 14
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ```help(ta.ma)```. Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""
