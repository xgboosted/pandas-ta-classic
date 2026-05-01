# -*- coding: utf-8 -*-
# Exponential Decay (EDECAY)
from typing import Any, Optional
from numpy import exp as npExp, maximum
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def edecay(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Exponential Decay (EDECAY)

    Multiplicative exponential decay that floors at the current close.
    Formula: result[i] = max(close[i], result[i-1] * exp(-1/length))
    tulipy name: EDECAY.
    """
    length = int(length) if length and length > 0 else 5
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    factor = npExp(-1.0 / length)
    arr = close.to_numpy(float)
    result = arr.copy()
    for i in range(1, len(arr)):
        result[i] = (
            arr[i]
            if arr[i] >= result[i - 1] * factor
            else max(arr[i], result[i - 1] * factor)
        )

    result = Series(result, index=close.index)

    # Offset
    result = apply_offset(result, offset)

    result.name = f"EDECAY_{length}"
    result.category = "trend"
    return result


edecay.__doc__ = """Exponential Decay (EDECAY)

At each bar the value decays multiplicatively from the previous bar,
floored at the current close price.
Formula: result[i] = max(close[i], result[i-1] * exp(-1/length))
tulipy name: EDECAY.

Args:
    close (pd.Series): Series of 'close' prices.
    length (int): Period. Default: 5.
    offset (int): Number of periods to offset the result. Default: 0.

Returns:
    pd.Series: Exponential decay series.
"""


edecay.__doc__ = """Exponential Decay (EDECAY)

Exponential variant of linear decay.  At each bar the value decays by
exp(-length) from the previous bar, floored at the current close price.

Equivalent to ta.decay(close, length, mode='exponential').
tulipy name: EDECAY.

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Decay period. Default: 5
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
