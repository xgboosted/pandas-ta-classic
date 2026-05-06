# -*- coding: utf-8 -*-
# Entropy (ENTROPY)
from typing import Any, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def entropy(
    close: Series,
    length: Optional[int] = None,
    base: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Entropy (ENTP)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    base = float(base) if base and base > 0 else 2.0
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Pure numpy for cross-version determinism.
    # Each window's probabilities share the same denominator (the window sum),
    # so they form a valid distribution and the result is proper Shannon entropy.
    values = close.values.astype(np.float64)
    n = len(values)
    result_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= length:
        windows = sliding_window_view(values, length)  # (n-length+1, length)
        window_sums = windows.sum(axis=1)  # (n-length+1,)
        valid = window_sums != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(valid[:, None], windows / window_sums[:, None], np.nan)
            p_term = -p * np.log(p) / np.log(base)
        # np.nansum treats 0*log(0)=NaN as 0, consistent with Shannon convention
        entropy_vals = np.where(valid, np.nansum(p_term, axis=1), np.nan)
        result_arr[length - 1 :] = entropy_vals
    entropy = Series(result_arr, index=close.index, dtype=np.float64)

    # Offset
    entropy = apply_offset(entropy, offset)

    entropy = apply_fill(entropy, **kwargs)

    # Name & Category
    entropy.name = f"ENTP_{length}"
    entropy.category = "statistics"

    return entropy


entropy.__doc__ = """Entropy (ENTP)

Introduced by Claude Shannon in 1948, entropy measures the unpredictability
of the data, or equivalently, of its average information. A die has higher
entropy (p=1/6) versus a coin (p=1/2).

Sources:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)

Calculation:
    Default Inputs:
        length=10, base=2

    P = close / SUM(close, length)
    E = SUM(-P * npLog(P) / npLog(base), length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    base (float): Logarithmic Base. Default: 2
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
