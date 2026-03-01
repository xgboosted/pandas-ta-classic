# -*- coding: utf-8 -*-
# Variable Index Dynamic Average (VIDYA)
from typing import Any, Optional
import numpy as np
from pandas import Series

npNaN = np.nan
from pandas_ta_classic.utils import _finalize, get_drift, get_offset, verify_series


def vidya(
    close: Series,
    length: Optional[int] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Variable Index Dynamic Average (VIDYA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    def _cmo(source: Series, n: int, d: int) -> Series:
        """Chande Momentum Oscillator (CMO) - Inlined to avoid circular import

        Note: This is inlined rather than imported from pandas_ta_classic.momentum.cmo
        to prevent a circular import issue:
        ma -> vidya -> cmo -> (momentum/__init__) -> apo -> ma
        """
        mom = source.diff(d)
        positive = mom.clip(lower=0)
        negative = mom.clip(upper=0).abs()
        pos_sum = positive.rolling(n).sum()
        neg_sum = negative.rolling(n).sum()
        return (pos_sum - neg_sum) / (pos_sum + neg_sum)

    # Calculate Result
    from pandas_ta_classic.utils._numba import _vidya_loop

    m = close.size
    alpha = 2 / (length + 1)
    abs_cmo = _cmo(close, length, drift).abs()
    cmo_arr = abs_cmo.to_numpy()
    c_arr = close.to_numpy()
    seed = float(c_arr[:length].mean())
    vidya_arr = _vidya_loop(c_arr, cmo_arr, m, length, alpha, seed)
    vidya = Series(vidya_arr, index=close.index)

    return _finalize(vidya, offset, f"VIDYA_{length}", "overlap", **kwargs)


vidya.__doc__ = """Variable Index Dynamic Average (VIDYA)

Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande. It is
similar to an Exponential Moving Average but it has a dynamically adjusted
lookback period dependent on relative price volatility as measured by Chande
Momentum Oscillator (CMO). When volatility is high, VIDYA reacts faster to
price changes. It is often used as moving average or trend identifier.

Sources:
    https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
    https://www.perfecttrendsystem.com/blog_mt4_2/en/vidya-indicator-for-mt4

Calculation:
    Default Inputs:
        length=10, adjust=False, sma=True
    if sma:
        sma_nth = close[0:length].sum() / length
        close[:length - 1] = np.nan
        close.iloc[length - 1] = sma_nth
    EMA = close.ewm(span=length, adjust=adjust).mean()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool, optional): Use adjust option for EMA calculation. Default: False
    sma (bool, optional): If True, uses SMA for initial value for EMA calculation. Default: True
    talib (bool): If True, uses TA-Libs implementation for CMO. Otherwise uses EMA version. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
