# McGinley Dynamic (MCGD)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int, _positive_inputs, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _mcgd_loop(c_arr, n, c, length):
    result = np.empty(n)
    result[0] = c_arr[0]
    for i in range(1, n):
        if result[i - 1] != 0:
            denom = c * length * (c_arr[i] / result[i - 1]) ** 4
            # At most the whole way to the price: below 1 the step overshot it,
            # and near zero the old 1e-10 floor sent the line to inf (#345).
            denom = max(denom, 1.0)
            result[i] = result[i - 1] + (c_arr[i] - result[i - 1]) / denom
        else:
            result[i] = c_arr[i]
    return result


@nan_on_short_input
@_positive_inputs("close")
@skip_leading_nan("close", interior=True)
def mcgd(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    c: float | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: McGinley Dynamic Indicator"""
    # Validate arguments
    length = _pos_int(length, 10, "length")
    c = _pos_float(c, 1, "c")
    if c > 1:
        raise ValueError(f"mcgd() c must be a number > 0 and <= 1, got {c!r}")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    c_arr = close.to_numpy(dtype=float)
    n = len(c_arr)
    result = _mcgd_loop(c_arr, n, c, length)
    mcg_ds = Series(result, index=close.index)

    # Offset
    mcg_ds = apply_offset(mcg_ds, offset)

    mcg_ds = apply_fill(mcg_ds, **kwargs)

    # Name & Category
    mcg_ds.name = f"MCGD_{length}"
    mcg_ds.category = "overlap"

    return mcg_ds


mcgd.__doc__ = """McGinley Dynamic Indicator

The McGinley Dynamic looks like a moving average line, yet it is actually a
smoothing mechanism for prices that minimizes price separation, price whipsaws,
and hugs prices much more closely. Because of the calculation, the Dynamic Line
speeds up in down markets as it follows prices yet moves more slowly in up
markets. The indicator was designed by John R. McGinley, a Certified Market
Technician and former editor of the Market Technicians Association's Journal
of Technical Analysis.

Sources:
    https://www.investopedia.com/articles/forex/09/mcginley-dynamic-indicator.asp

Calculation:
    Default Inputs:
        length=10
        offset=0
        c=1

    MD[0] = close[0]
    denom = max(c * length * (close[i] / MD[i - 1]) ** 4, 1)
    MD[i] = MD[i - 1] + (close[i] - MD[i - 1]) / denom

    The bound of 1 keeps a step from passing the price. A close at or below
    zero reads NaN, with one warning, and the line restarts after it.

Args:
    close (pd.Series): Series of 'close's
    length (int): Indicator's period. Default: 10
    offset (int): Number of periods to offset the result. Default: 0
    c (float): Multiplier for the denominator, sometimes set to 0.6. Default: 1

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
