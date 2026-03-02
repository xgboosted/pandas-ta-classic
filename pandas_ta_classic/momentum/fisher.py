# -*- coding: utf-8 -*-
# Fisher Transform (FISHER)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.overlap.hl2 import hl2
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_offset,
    high_low_range,
    verify_series,
)


def fisher(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    signal: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Fisher Transform (FISHT)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 9
    signal = int(signal) if signal and signal > 0 else 1
    _length = max(length, signal)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    hl2_ = hl2(high, low)
    highest_hl2 = hl2_.rolling(length).max()
    lowest_hl2 = hl2_.rolling(length).min()

    hlr = high_low_range(highest_hl2, lowest_hl2)
    hlr[hlr < 0.001] = 0.001

    position = ((hl2_ - lowest_hl2) / hlr) - 0.5

    from pandas_ta_classic.utils._numba import _fisher_loop

    pos_arr = position.to_numpy()
    m = high.size
    result = _fisher_loop(pos_arr, m, length)
    fisher = Series(result, index=high.index)
    signalma = fisher.shift(signal)

    # Offset + Name + Category + DataFrame
    _props = f"_{length}_{signal}"
    return _build_dataframe(
        {f"FISHERT{_props}": fisher, f"FISHERTs{_props}": signalma},
        f"FISHERT{_props}",
        "momentum",
        offset,
        **kwargs,
    )


fisher.__doc__ = """Fisher Transform (FISHT)

Attempts to identify significant price reversals by normalizing prices over a
user-specified number of periods. A reversal signal is suggested when the the
two lines cross.

Sources:
    TradingView (Correlation >99%)

Calculation:
    Default Inputs:
        length=9, signal=1
    HL2 = hl2(high, low)
    HHL2 = HL2.rolling(length).max()
    LHL2 = HL2.rolling(length).min()

    HLR = HHL2 - LHL2
    HLR[HLR < 0.001] = 0.001

    position = ((HL2 - LHL2) / HLR) - 0.5

    v = 0
    m = high.size
    FISHER = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        v = 0.66 * position[i] + 0.67 * v
        if v < -0.99: v = -0.999
        if v >  0.99: v =  0.999
        FISHER.append(0.5 * (nplog((1 + v) / (1 - v)) + FISHER[i - 1]))

    SIGNAL = FISHER.shift(signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    length (int): Fisher period. Default: 9
    signal (int): Fisher Signal period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
