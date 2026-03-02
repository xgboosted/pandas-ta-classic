# Even Better Sine Wave (EBSW)
from typing import Any, Optional
import numpy as np
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sin as npSin
from numpy import sqrt as npSqrt
from pandas import Series

npNaN = np.nan
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def ebsw(
    close: Series,
    length: Optional[int] = None,
    bars: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Even Better SineWave (EBSW)"""
    # Validate arguments
    length = int(length) if length and length > 38 else 40
    bars = int(bars) if bars and bars > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Precompute filter constants (depend only on length/bars, not on bar index).
    alpha1 = (1 - npSin(360 / length)) / npCos(360 / length)
    a1 = npExp(-npSqrt(2) * npPi / bars)
    b1 = 2 * a1 * npCos(npSqrt(2) * 180 / bars)
    c2 = b1
    c3 = -1 * a1 * a1
    c1 = 1 - c2 - c3

    # Calculate Result
    from pandas_ta_classic.utils._numba import _ebsw_loop

    c_arr = close.to_numpy()
    m = close.size
    result = _ebsw_loop(c_arr, m, length, alpha1, c1, c2, c3)

    ebsw = Series(result, index=close.index)

    return _finalize(ebsw, offset, f"EBSW_{length}_{bars}", "cycles", **kwargs)


ebsw.__doc__ = """Even Better SineWave (EBSW) *beta*

This indicator measures market cycles and uses a low pass filter to remove noise.
Its output is bound signal between -1 and 1 and the maximum length of a detected
trend is limited by its length input.

Written by rengel8 for Pandas TA based on a publication at 'prorealcode.com' and
a book by J.F.Ehlers.

* This implementation seems to be logically limited. It would make sense to
implement exactly the version from prorealcode and compare the behaviour.


Sources:
    https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
    J.F.Ehlers 'Cycle Analytics for Traders', 2014

Calculation:
    refer to 'sources' or implementation

Args:
    close (pd.Series): Series of 'close's
    length (int): It's max cycle/trend period. Values between 40-48 work like
        expected with minimum value: 39. Default: 40.
    bars (int): Period of low pass filtering. Default: 10
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
