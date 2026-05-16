# -*- coding: utf-8 -*-
# Normalized Average True Range (NATR)
from typing import Any, Optional
import numpy as np
from pandas import Series
from .atr import atr
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def natr(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Normalized Average True Range (NATR)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    # BREAKING CHANGE: default changed from "ema" to "rma" to match TA-Lib's
    # Wilder's smoothing. Callers relying on the old EMA default will see
    # different output unless they pass mamode="ema" explicitly.
    mamode = mamode if isinstance(mamode, str) else "rma"
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import NATR

        natr = NATR(high, low, close, length)
    else:
        # Direct Wilder loop to match TA-Lib's exact floating-point operations.
        # Seeding both ATR and NATR from the same SMA value at bar `length`
        # avoids the rounding introduced by multiplying a pandas Series ATR
        # by `scalar/close` after the fact.
        h_arr = high.to_numpy(dtype=float)
        l_arr = low.to_numpy(dtype=float)
        c_arr = close.to_numpy(dtype=float)
        m = len(c_arr)
        tr_arr = np.full(m, np.nan)
        for i in range(1, m):
            tr_arr[i] = max(
                h_arr[i] - l_arr[i],
                abs(h_arr[i] - c_arr[i - 1]),
                abs(l_arr[i] - c_arr[i - 1]),
            )
        atr_arr = np.full(m, np.nan)
        if m > length:
            atr_val = tr_arr[1 : length + 1].mean()
            atr_arr[length] = atr_val
            for i in range(length + 1, m):
                atr_val = atr_val + (tr_arr[i] - atr_val) / length
                atr_arr[i] = atr_val
        natr = Series(scalar * atr_arr / c_arr, index=close.index)

    # Offset
    natr = apply_offset(natr, offset)

    natr = apply_fill(natr, **kwargs)

    # Name and Categorize it
    natr.name = f"NATR_{length}"
    natr.category = "volatility"

    return natr


natr.__doc__ = """Normalized Average True Range (NATR)

Normalized Average True Range attempt to normalize the average true range.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

Calculation:
    Default Inputs:
        length=20
    ATR = Average True Range
    NATR = (100 / close) * ATR(high, low, close)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 20
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ```help(ta.ma)```. Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""
