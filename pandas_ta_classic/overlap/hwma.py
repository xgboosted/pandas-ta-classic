# -*- coding: utf-8 -*-
# Holt-Winter Moving Average (HWMA)
from typing import Any, Optional
import numpy as np
from pandas import Series
from pandas_ta_classic.utils import get_offset, verify_series


def _hwma_loop(c_arr, m, na, nb, nc):
    result = np.empty(m)
    last_a = 0.0
    last_v = 0.0
    last_f = c_arr[0]
    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * c_arr[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result[i] = F + V + 0.5 * A
        last_a = A
        last_f = F
        last_v = V
    return result


def hwma(
    close: Series,
    na: Optional[float] = None,
    nb: Optional[float] = None,
    nc: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Holt-Winter Moving Average"""
    # Validate Arguments
    na = float(na) if na and na > 0 and na < 1 else 0.2
    nb = float(nb) if nb and nb > 0 and nb < 1 else 0.1
    nc = float(nc) if nc and nc > 0 and nc < 1 else 0.1
    close = verify_series(close)
    offset = get_offset(offset)

    # Calculate Result
    m = close.size
    c_arr = close.to_numpy(dtype=float)
    result = _hwma_loop(c_arr, m, na, nb, nc)
    hwma = Series(result, index=close.index)

    # Offset
    if offset != 0:
        hwma = hwma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        hwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                hwma.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                hwma.bfill(inplace=True)

    # Name & Category
    suffix = f"{na}_{nb}_{nc}"
    hwma.name = f"HWMA_{suffix}"
    hwma.category = "overlap"

    return hwma


hwma.__doc__ = """HWMA (Holt-Winter Moving Average)

Indicator HWMA (Holt-Winter Moving Average) is a three-parameter moving average
by the Holt-Winter method; the three parameters should be selected to obtain a
forecast.

This version has been implemented for Pandas TA by rengel8 based
on a publication for MetaTrader 5.

Sources:
    https://www.mql5.com/en/code/20856

Calculation:
    HWMA[i] = F[i] + V[i] + 0.5 * A[i]
    where..
    F[i] = (1-na) * (F[i-1] + V[i-1] + 0.5 * A[i-1]) + na * Price[i]
    V[i] = (1-nb) * (V[i-1] + A[i-1]) + nb * (F[i] - F[i-1])
    A[i] = (1-nc) * A[i-1] + nc * (V[i] - V[i-1])

Args:
    close (pd.Series): Series of 'close's
    na (float): Smoothed series parameter (from 0 to 1). Default: 0.2
    nb (float): Trend parameter (from 0 to 1). Default: 0.1
    nc (float): Seasonality parameter (from 0 to 1). Default: 0.1
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: hwma
"""
