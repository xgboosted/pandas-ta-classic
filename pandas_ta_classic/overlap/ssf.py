# -*- coding: utf-8 -*-
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sqrt as npSqrt
from pandas_ta_classic.utils import get_offset, verify_series
from pandas_ta_classic.utils._numba import NUMBA_AVAILABLE, ssf_numba_core


def ssf(close, length=None, poles=None, offset=None, use_numba=True, **kwargs):
    """Indicator: Ehler's Super Smoother Filter (SSF)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    poles = int(poles) if poles in [2, 3] else 2
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return

    # Calculate Result
    m = close.size

    if poles == 3:
        x = npPi / length
        a0 = npExp(-x)
        b0 = 2 * a0 * npCos(npSqrt(3) * x)
        c0 = a0 * a0

        c4 = c0 * c0
        c3 = -c0 * (1 + b0)
        c2 = c0 + b0
        c1 = 1 - c2 - c3 - c4

        # Use Numba-optimized implementation if available and requested
        if use_numba and NUMBA_AVAILABLE:
            result_values = ssf_numba_core(close.values, c1, c2, c3, c4, poles)
            ssf = close.copy()
            ssf[:] = result_values
        else:
            # Fall back to pure Python implementation
            ssf = close.copy()
            for i in range(0, m):
                ssf.iloc[i] = (
                    c1 * close.iloc[i]
                    + c2 * ssf.iloc[i - 1]
                    + c3 * ssf.iloc[i - 2]
                    + c4 * ssf.iloc[i - 3]
                )

    else:  # poles == 2
        x = npPi * npSqrt(2) / length
        a0 = npExp(-x)
        a1 = -a0 * a0
        b1 = 2 * a0 * npCos(x)
        c1 = 1 - a1 - b1

        # Use Numba-optimized implementation if available and requested
        if use_numba and NUMBA_AVAILABLE:
            result_values = ssf_numba_core(close.values, c1, b1, a1, 0.0, poles)
            ssf = close.copy()
            ssf[:] = result_values
        else:
            # Fall back to pure Python implementation
            ssf = close.copy()
            for i in range(0, m):
                ssf.iloc[i] = (
                    c1 * close.iloc[i] + b1 * ssf.iloc[i - 1] + a1 * ssf.iloc[i - 2]
                )

    # Offset
    if offset != 0:
        ssf = ssf.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ssf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                ssf.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                ssf.bfill(inplace=True)

    # Name & Category
    ssf.name = f"SSF_{length}_{poles}"
    ssf.category = "overlap"

    return ssf


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
