# Choppiness Index (CHOP)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_int, nan_on_short_input
from pandas_ta_classic.volatility.atr import atr


@nan_on_short_input
def chop(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    atr_length: int | None = None,
    ln: bool | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Choppiness Index (CHOP)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    atr_length = _pos_int(atr_length, 1, "atr_length")
    ln = _bool_param(ln, False, "ln")
    scalar = _number(scalar, 100, "scalar")
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    # A strategy-wide drift (df.ta.strategy(..., drift=N)) has no meaning here: the
    # parameter was removed in 0.9.0. Drop it rather than forward it to apply_fill.
    kwargs.pop("drift", None)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    diff = high.rolling(length).max() - low.rolling(length).min()

    atr_ = atr(high=high, low=low, close=close, length=atr_length)
    if atr_ is None:
        return None
    atr_sum = atr_.rolling(length).sum()

    if ln:
        chop = scalar * (np.log(atr_sum) - np.log(diff)) / np.log(length)
    else:
        chop = scalar * (np.log10(atr_sum) - np.log10(diff)) / np.log10(length)

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
        length=14, scalar=100
    HH = high.rolling(length).max()
    LL = low.rolling(length).min()

    ATR_SUM = SUM(ATR, length)
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
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
