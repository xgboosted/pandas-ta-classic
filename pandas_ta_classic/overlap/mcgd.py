# -*- coding: utf-8 -*-
# McGinley Dynamic (MCGD)
from typing import Any, Optional
import numpy as np
import pandas as pd
from pandas import Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _mcgd_loop(c_arr, n, c, length):
    result = np.empty(n)
    result[0] = c_arr[0]
    for i in range(1, n):
        if result[i - 1] != 0:
            denom = c * length * (c_arr[i] / result[i - 1]) ** 4
            if denom < 1e-10:
                denom = 1e-10
            result[i] = result[i - 1] + (c_arr[i] - result[i - 1]) / denom
        else:
            result[i] = c_arr[i]
    return result


def mcgd(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    c: Optional[float] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: McGinley Dynamic Indicator"""
    # Validate arguments
    length = int(length) if length and length > 0 else 10
    c = float(c) if c and 0 < c <= 1 else 1
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

    def mcg_(series):
        denom = (constant * length * (series.iloc[1] / series.iloc[0]) ** 4)
        series.iloc[1] = (series.iloc[0] + ((series.iloc[1] - series.iloc[0]) / denom))
        return series.iloc[1]
    mcg_cell = close[0:].rolling(2, min_periods=2).apply(mcg_, raw=False)
    mcg_ds = pd.concat([close[:1], mcg_cell[1:]])

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
