# -*- coding: utf-8 -*-
# Money Flow Index (MFI)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def mfi(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Money Flow Index (MFI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import MFI

        mfi = MFI(high, low, close, volume, length)
    else:
        h_arr = high.to_numpy(dtype=float)
        l_arr = low.to_numpy(dtype=float)
        c_arr = close.to_numpy(dtype=float)
        v_arr = volume.to_numpy(dtype=float)
        tp = (h_arr + l_arr + c_arr) / 3.0
        rmf = tp * v_arr
        m = len(tp)

        # Direct sliding-window loop matching TA-Lib's exact FP operations.
        # Initial window: bars 1..length (bar 0 skipped — no previous bar for diff).
        psum = 0.0
        nsum = 0.0
        for i in range(1, length + 1):
            if tp[i] > tp[i - 1]:
                psum += rmf[i]
            elif tp[i] < tp[i - 1]:
                nsum += rmf[i]

        mfi_arr = np.full(m, np.nan)
        total = psum + nsum
        mfi_arr[length] = 100.0 * psum / total if total >= 1.0 else 0.0

        trailing = 1
        for i in range(length + 1, m):
            if tp[i] > tp[i - 1]:
                psum += rmf[i]
            elif tp[i] < tp[i - 1]:
                nsum += rmf[i]
            if tp[trailing] > tp[trailing - 1]:
                psum -= rmf[trailing]
            elif tp[trailing] < tp[trailing - 1]:
                nsum -= rmf[trailing]
            trailing += 1
            total = psum + nsum
            mfi_arr[i] = 100.0 * psum / total if total >= 1.0 else 0.0

        mfi = Series(mfi_arr, index=close.index)

    # Offset
    mfi = apply_offset(mfi, offset)

    mfi = apply_fill(mfi, **kwargs)

    # Name and Categorize it
    mfi.name = f"MFI_{length}"
    mfi.category = "volume"

    return mfi


mfi.__doc__ = """Money Flow Index (MFI)

Money Flow Index is an oscillator indicator that is used to measure buying and
selling pressure by utilizing both price and volume.

Sources:
    https://www.tradingview.com/wiki/Money_Flow_(MFI)

Calculation:
    Default Inputs:
        length=14, drift=1
    tp = typical_price = hlc3 = (high + low + close) / 3
    rmf = raw_money_flow = tp * volume

    pmf = pos_money_flow = SUM(rmf, length) if tp.diff(drift) > 0 else 0
    nmf = neg_money_flow = SUM(rmf, length) if tp.diff(drift) < 0 else 0

    MFR = money_flow_ratio = pmf / nmf
    MFI = money_flow_index = 100 * pmf / (pmf + nmf)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The sum period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
