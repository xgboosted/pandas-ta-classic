# -*- coding: utf-8 -*-
# Kurtosis (KURTOSIS)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import (
    _get_min_periods,
    _finalize,
    get_offset,
    np_rolling_moments,
    verify_series,
)


def kurtosis(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Kurtosis"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy rolling excess kurtosis (Fisher) for cross-version determinism.
    # G2 = n(n+1)(n-1)·M4 / ((n-2)(n-3)·M2²) - 3(n-1)²/((n-2)(n-3))
    m2, m4 = np_rolling_moments(close.values, length, 2, 4)
    nf = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = nf * (nf + 1) * (nf - 1) * m4
        denom = (nf - 2) * (nf - 3) * m2**2
        adj = 3.0 * (nf - 1) ** 2 / ((nf - 2) * (nf - 3))
        result = numer / denom - adj

    kurtosis = Series(result, index=close.index, dtype=np.float64)

    return _finalize(kurtosis, offset, f"KURT_{length}", "statistics", **kwargs)


kurtosis.__doc__ = """Rolling Kurtosis

Sources:

Calculation:
    Default Inputs:
        length=30
    KURTOSIS = close.rolling(length).kurt()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
