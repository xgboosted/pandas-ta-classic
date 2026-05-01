# -*- coding: utf-8 -*-
# Skew (SKEW)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
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
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy rolling skewness (adjusted Fisher-Pearson) for cross-version
    # determinism.
    m2, m3 = np_rolling_moments(close.values, length, 2, 3, min_periods=min_periods)
    # n_eff[i] is the actual window size at position i (scalar for the common
    # case where min_periods == length).
    if min_periods < length:
        n_eff = np.full(len(close), np.float64(length))
        for pos in range(min_periods - 1, min(length - 1, len(close))):
            n_eff[pos] = pos + 1
    else:
        n_eff = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = n_eff * np.sqrt(n_eff - 1) / (n_eff - 2) * m3 / m2**1.5
    skew = Series(result, index=close.index, dtype=np.float64)

    # Offset
    skew = apply_offset(skew, offset)

    skew = apply_fill(skew, **kwargs)

    # Name & Category
    skew.name = f"SKEW_{length}"
    skew.category = "statistics"

    return skew


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
