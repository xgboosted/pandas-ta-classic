# Chande Forecast Oscillator (CFO)
from typing import Annotated, Any

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import Bound, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("close")
def cfo(
    close: Series,
    length: Annotated[int, Bound(gt=1)] = 9,
    scalar: Annotated[float, Bound()] = 100,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Chande Forecast Oscillator (CFO)"""
    # Validate Arguments
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Finding linear regression of Series
    _linreg = linreg(close, length=length, tsf=True)
    if _linreg is None:
        return None
    cfo = scalar * (close - _linreg)
    cfo /= close

    # Offset
    cfo = apply_offset(cfo, offset)

    cfo = apply_fill(cfo, **kwargs)

    # Name and Categorize it
    cfo.name = f"CFO_{length}"
    cfo.category = "momentum"

    return cfo


cfo.__doc__ = """Chande Forecast Oscillator (CFO)

The Forecast Oscillator calculates the percentage difference between the actual
price and the Time Series Forecast (the endpoint of a linear regression line).

Sources:
    https://www.fmlabs.com/reference/default.htm?url=ForecastOscillator.htm

Calculation:
    Default Inputs:
        length=9, scalar=100
    LINREG = Linear Regression

    CFO = scalar * (close - LINERREG(length, tdf=True)) / close

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 9
    scalar (float): How much to magnify. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
