# Even Better Sine Wave (EBSW)
from typing import Any, Optional
import numpy as np
from pandas import Series


from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _ebsw_nb(close, length, bars):
    m = close.size
    result = np.full(m, np.nan)
    if length - 1 < m:
        result[length - 1] = 0.0

    # HighPass and SuperSmoother coefficients are constant across bars
    alpha1 = (1 - np.sin(360 / length)) / np.cos(360 / length)
    a1 = np.exp(-np.sqrt(2) * np.pi / bars)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * 180 / bars)
    c2 = b1
    c3 = -1 * a1 * a1
    c1 = 1 - c2 - c3

    lastClose = 0.0
    lastHP = 0.0
    fh0 = 0.0  # FilterHist[0] (older)
    fh1 = 0.0  # FilterHist[1] (recent)

    for i in range(length, m):
        HP = 0.5 * (1 + alpha1) * (close[i] - lastClose) + alpha1 * lastHP
        Filt = c1 * (HP + lastHP) / 2 + c2 * fh1 + c3 * fh0

        # 3 Bar average of Wave amplitude and power
        Wave = (Filt + fh1 + fh0) / 3
        Pwr = (Filt * Filt + fh1 * fh1 + fh0 * fh0) / 3

        # Normalize the Average Wave to Square Root of the Average Power
        Wave = Wave / np.sqrt(Pwr) if Pwr > 0 else 0.0

        # update storage
        fh0 = fh1
        fh1 = Filt
        lastHP = HP
        lastClose = close[i]
        result[i] = Wave

    return result


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

    result = _ebsw_nb(close.to_numpy(dtype=float), length, bars)
    ebsw = Series(result, index=close.index)

    # Offset
    ebsw = apply_offset(ebsw, offset)

    ebsw = apply_fill(ebsw, **kwargs)

    # Name and Categorize it
    ebsw.name = f"EBSW_{length}_{bars}"
    ebsw.category = "cycles"

    return ebsw


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
