# -*- coding: utf-8 -*-
# Holt-Winter Moving Average (HWMA)
from typing import Any, Optional
import numpy as np
from pandas import Series
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


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

    if close is None:
        return None

    # Calculate Result
    from pandas_ta_classic.utils._numba import _hwma_loop

    m = close.size
    c_arr = close.to_numpy()
    result_arr = _hwma_loop(c_arr, m, na, nb, nc)

    hwma = Series(result_arr, index=close.index)

    return _finalize(hwma, offset, f"HWMA_{na}_{nb}_{nc}", "overlap", **kwargs)


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
