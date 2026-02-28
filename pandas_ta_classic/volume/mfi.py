# -*- coding: utf-8 -*-
# Money Flow Index (MFI)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.utils import apply_offset, get_drift, get_offset, verify_series


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
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MFI

        mfi = MFI(high, low, close, volume, length)
    else:
        typical_price = hlc3(high=high, low=low, close=close)
        raw_money_flow = typical_price * volume

        tdf = DataFrame({"diff": 0, "rmf": raw_money_flow, "+mf": 0.0, "-mf": 0.0})

        tdf.loc[(typical_price.diff(drift) > 0), "diff"] = 1
        tdf.loc[tdf["diff"] == 1, "+mf"] = raw_money_flow

        tdf.loc[(typical_price.diff(drift) < 0), "diff"] = -1
        tdf.loc[tdf["diff"] == -1, "-mf"] = raw_money_flow

        psum = tdf["+mf"].rolling(length).sum()
        nsum = tdf["-mf"].rolling(length).sum()
        tdf["mr"] = psum / nsum
        mfi = 100 * psum / (psum + nsum)
        tdf["mfi"] = mfi

    # Offset
    mfi = apply_offset(mfi, offset, **kwargs)

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
