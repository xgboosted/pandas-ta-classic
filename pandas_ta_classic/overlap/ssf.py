# -*- coding: utf-8 -*-
# Super Smoother Filter (SSF)
from typing import Any, Optional
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sqrt as npSqrt
from pandas import Series
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def ssf(
    close: Series,
    length: Optional[int] = None,
    poles: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Ehler's Super Smoother Filter (SSF)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    poles = int(poles) if poles in [2, 3] else 2
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    from pandas_ta_classic.utils._numba import _ssf2_loop, _ssf3_loop

    m = close.size
    c_arr = close.to_numpy(dtype=float)
    ssf_arr = c_arr.copy()

    if poles == 3:
        x = npPi / length
        a0 = npExp(-x)
        b0 = 2 * a0 * npCos(npSqrt(3) * x)
        c0 = a0 * a0

        c4 = c0 * c0
        c3 = -c0 * (1 + b0)
        c2 = c0 + b0
        c1 = 1 - c2 - c3 - c4

        ssf_arr = _ssf3_loop(c_arr, ssf_arr, m, c1, c2, c3, c4)

    else:  # poles == 2
        x = npPi * npSqrt(2) / length
        a0 = npExp(-x)
        a1 = -a0 * a0
        b1 = 2 * a0 * npCos(x)
        c1 = 1 - a1 - b1

        ssf_arr = _ssf2_loop(c_arr, ssf_arr, m, c1, b1, a1)

    ssf = Series(ssf_arr, index=close.index)

    return _finalize(ssf, offset, f"SSF_{length}_{poles}", "overlap", **kwargs)


ssf.__doc__ = """Ehler's Super Smoother Filter (SSF) © 2013

John F. Ehlers's solution to reduce lag and remove aliasing noise with his
research in aerospace analog filter design. This indicator comes with two
versions determined by the keyword poles. By default, it uses two poles but
there is an option for three poles. Since SSF is a (Resursive) Digital Filter,
the number of poles determine how many prior recursive SSF bars to include in
the design of the filter. So two poles uses two prior SSF bars and three poles
uses three prior SSF bars for their filter calculations.

Sources:
    http://www.stockspotter.com/files/PredictiveIndicators.pdf
    https://www.tradingview.com/script/VdJy0yBJ-Ehlers-Super-Smoother-Filter/
    https://www.mql5.com/en/code/588
    https://www.mql5.com/en/code/589

Calculation:
    Default Inputs:
        length=10, poles=[2, 3]

    See the source code or Sources listed above.

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    poles (int): The number of poles to use, either 2 or 3. Default: 2
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
