# -*- coding: utf-8 -*-
# Quantile (QUANTILE)
from typing import Any, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series

from pandas_ta_classic.utils import (
    _get_min_periods,
    _finalize,
    get_offset,
    verify_series,
)


def quantile(
    close: Series,
    length: Optional[int] = None,
    q: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Quantile"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = _get_min_periods(kwargs, length)
    q = float(q) if q and q > 0 and q < 1 else 0.5
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy for cross-version determinism.
    values = close.values.astype(np.float64)
    windows = sliding_window_view(values, length)
    qtl = np.quantile(windows, q, axis=1)

    result: np.ndarray = np.empty(len(values), dtype=np.float64)
    result[: length - 1] = np.nan
    result[length - 1 :] = qtl

    quantile = Series(result, index=close.index, dtype=np.float64)

    return _finalize(quantile, offset, f"QTL_{length}_{q}", "statistics", **kwargs)


quantile.__doc__ = """Rolling Quantile

Sources:

Calculation:
    Default Inputs:
        length=30, q=0.5
    QUANTILE = close.rolling(length).quantile(q)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    q (float): The quantile. Default: 0.5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
