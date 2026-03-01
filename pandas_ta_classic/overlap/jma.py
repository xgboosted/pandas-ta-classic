# -*- coding: utf-8 -*-
# Jurik Moving Average (JMA)
from math import log as _log, pow as _pow, sqrt as _sqrt
from typing import Any, Optional, Union
import numpy as np
from numpy import zeros_like as npZeroslike
from pandas import Series

npNaN = np.nan
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def jma(
    close: Series,
    length: Optional[Union[int, float]] = None,
    phase: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Jurik Moving Average (JMA)"""
    # Validate Arguments
    _length = int(length) if length and length > 0 else 7
    phase = float(phase) if phase and phase != 0 else 0
    close = verify_series(close, _length)
    offset = get_offset(offset)
    if close is None:
        return None

    # Static variables
    sum_length = 10
    length = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((_log(_sqrt(length)) / _log(2.0)) + 2.0, 0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * _sqrt(length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)
    r_volty_max = _pow(length1, 1 / pow1)

    # Calculate Result
    from pandas_ta_classic.utils._numba import _jma_loop

    c_arr = close.to_numpy()
    m = close.shape[0]
    jma = _jma_loop(
        c_arr, m, sum_length, length, pr, length1, pow1, bet, beta, r_volty_max
    )

    # Remove initial lookback data and convert to pandas frame
    jma[0 : _length - 1] = npNaN
    jma = Series(jma, index=close.index)

    return _finalize(jma, offset, f"JMA_{_length}_{phase}", "overlap", **kwargs)


jma.__doc__ = """Jurik Moving Average Average (JMA)

Mark Jurik's Moving Average (JMA) attempts to eliminate noise to see the "true"
underlying activity. It has extremely low lag, is very smooth and is responsive
to market gaps.

Sources:
    https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    https://www.prorealcode.com/prorealtime-indicators/jurik-volatility-bands/

Calculation:
    Default Inputs:
        length=7, phase=0

Args:
    close (pd.Series): Series of 'close's
    length (int): Period of calculation. Default: 7
    phase (float): How heavy/light the average is [-100, 100]. Default: 0
    offset (int): How many lengths to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
