# -*- coding: utf-8 -*-
# Choppiness Index (CHOP)
from typing import Any, Optional
from numpy import log10 as npLog10
from numpy import log as npLn
from pandas import Series
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def chop(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    atr_length: Optional[int] = None,
    ln: Optional[bool] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Choppiness Index (CHOP)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    atr_length = int(atr_length) if atr_length is not None and atr_length > 0 else 1
    ln = bool(ln) if isinstance(ln, bool) else False
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    diff = high.rolling(length).max() - low.rolling(length).min()

    atr_ = atr(high=high, low=low, close=close, length=atr_length)
    if atr_ is None:
        return None
    atr_sum = atr_.rolling(length).sum()

    chop = scalar
    if ln:
        chop *= (npLn(atr_sum) - npLn(diff)) / npLn(length)
    else:
        chop *= (npLog10(atr_sum) - npLog10(diff)) / npLog10(length)

    # Offset
    chop = apply_offset(chop, offset)

    chop = apply_fill(chop, **kwargs)

    # Name and Categorize it
    chop.name = f"CHOP{'ln' if ln else ''}_{length}_{atr_length}_{scalar}"
    chop.category = "trend"

    return chop


chop.__doc__ = """Choppiness Index (CHOP)

The Choppiness Index was created by Australian commodity trader
E.W. Dreiss and is designed to determine if the market is choppy
(trading sideways) or not choppy (trading within a trend in either
direction). Values closer to 100 implies the underlying is choppier
whereas values closer to 0 implies the underlying is trending.

Sources:
    https://www.tradingview.com/scripts/choppinessindex/
    https://www.motivewave.com/studies/choppiness_index.htm

Calculation:
    Default Inputs:
        length=14, scalar=100, drift=1
    HH = high.rolling(length).max()
    LL = low.rolling(length).min()

    ATR_SUM = SUM(ATR(drift), length)
    CHOP = scalar * (LOG10(ATR_SUM) - LOG10(HH - LL))
    CHOP /= LOG10(length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    atr_length (int): Length for ATR. Default: 1
    ln (bool): If True, uses ln otherwise log10. Default: False
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
