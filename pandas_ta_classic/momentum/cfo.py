# Chande Forecast Oscillator (CFO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import _finalize, get_drift, get_offset, verify_series


def cfo(
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Chande Forcast Oscillator (CFO)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 9
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    # Finding linear regression of Series
    cfo = scalar * (close - linreg(close, length=length, tsf=True))
    cfo /= close

    return _finalize(cfo, offset, f"CFO_{length}", "momentum", **kwargs)


cfo.__doc__ = """Chande Forcast Oscillator (CFO)

The Forecast Oscillator calculates the percentage difference between the actual
price and the Time Series Forecast (the endpoint of a linear regression line).

Sources:
    https://www.fmlabs.com/reference/default.htm?url=ForecastOscillator.htm

Calculation:
    Default Inputs:
        length=9, drift=1, scalar=100
    LINREG = Linear Regression

    CFO = scalar * (close - LINERREG(length, tdf=True)) / close

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 9
    scalar (float): How much to magnify. Default: 100
    drift (int): The short period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
