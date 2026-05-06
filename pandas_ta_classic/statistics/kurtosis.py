# -*- coding: utf-8 -*-
# Kurtosis (KURTOSIS)
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


def kurtosis(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Kurtosis"""
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

    # Pure numpy rolling excess kurtosis (Fisher) for cross-version determinism.
    m2, m4 = np_rolling_moments(close.values, length, 2, 4, min_periods=min_periods)
    # n_eff[i] is the actual window size used at position i.  When
    # min_periods == length (the default) every position uses length, so a
    # scalar is sufficient and avoids the array-allocation overhead.
    if min_periods < length:
        n_eff = np.full(len(close), np.float64(length))
        for pos in range(min_periods - 1, min(length - 1, len(close))):
            n_eff[pos] = pos + 1
    else:
        n_eff = np.float64(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = n_eff * (n_eff + 1) * (n_eff - 1) * m4
        denom = (n_eff - 2) * (n_eff - 3) * m2**2
        adj = 3.0 * (n_eff - 1) ** 2 / ((n_eff - 2) * (n_eff - 3))
        result = numer / denom - adj
    kurtosis = Series(result, index=close.index, dtype=np.float64)

    # Offset
    kurtosis = apply_offset(kurtosis, offset)

    kurtosis = apply_fill(kurtosis, **kwargs)

    # Name & Category
    kurtosis.name = f"KURT_{length}"
    kurtosis.category = "statistics"

    return kurtosis


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
