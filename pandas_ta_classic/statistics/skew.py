# -*- coding: utf-8 -*-
# Skew (SKEW)
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


def skew(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Skew"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy rolling skewness (adjusted Fisher-Pearson) for cross-version
    # determinism.
    # G1 = n·√(n−1) / (n−2) · M3 / M2^(3/2)
    m2, m3 = np_rolling_moments(close.values, length, 2, 3)
    nf = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = nf * np.sqrt(nf - 1) / (nf - 2) * m3 / m2**1.5

    skew = Series(result, index=close.index, dtype=np.float64)

    return _finalize(skew, offset, f"SKEW_{length}", "statistics", **kwargs)


skew.__doc__ = """Rolling Skew

Sources:

Calculation:
    Default Inputs:
        length=30
    SKEW = close.rolling(length).skew()

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
