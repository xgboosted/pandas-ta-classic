# Quantile (QUANTILE)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int, nan_on_short_input


@nan_on_short_input
def quantile(
    close: Series,
    length: int | None = None,
    q: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Quantile"""
    # Validate Arguments
    length = _pos_int(length, 30, "length")
    min_periods = _pos_int(kwargs.get("min_periods"), length, "min_periods", gt=None, ge=0)
    q = _pos_float(q, 0.5, "q", lt=1)
    close = verify_series(close, max(length, min_periods))
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy for cross-version determinism.
    values = close.values.astype(np.float64)
    n = len(values)
    result_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= length:
        windows = np.lib.stride_tricks.sliding_window_view(values, length)
        result_arr[length - 1 :] = np.quantile(windows, q, axis=1)
    if min_periods < length:
        for pos in range(min_periods - 1, min(length - 1, n)):
            result_arr[pos] = np.quantile(values[: pos + 1], q)
    quantile = Series(result_arr, index=close.index, dtype=np.float64)

    # Offset
    quantile = apply_offset(quantile, offset)

    quantile = apply_fill(quantile, **kwargs)

    # Name & Category
    quantile.name = f"QTL_{length}_{q}"
    quantile.category = "statistics"

    return quantile


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
