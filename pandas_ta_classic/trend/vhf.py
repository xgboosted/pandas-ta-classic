# -*- coding: utf-8 -*-
# Vertical Horizontal Filter (VHF)
from typing import Any, Optional
from numpy import fabs as npFabs
from pandas import Series
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def vhf(
    close: Series,
    length: Optional[int] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Vertical Horizontal Filter (VHF)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 28
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    hcp = close.rolling(length).max()
    lcp = close.rolling(length).min()
    diff = npFabs(close.diff(drift))
    vhf = npFabs(non_zero_range(hcp, lcp)) / diff.rolling(length).sum()

    # Offset
    vhf = apply_offset(vhf, offset)

    vhf = apply_fill(vhf, **kwargs)

    # Name and Categorize it
    vhf.name = f"VHF_{length}"
    vhf.category = "trend"

    return vhf


vhf.__doc__ = """Vertical Horizontal Filter (VHF)

VHF was created by Adam White to identify trending and ranging markets.

Sources:
    https://www.incrediblecharts.com/indicators/vertical_horizontal_filter.php

Calculation:
    Default Inputs:
        length = 28
    HCP = Highest Close Price in Period
    LCP = Lowest Close Price in Period
    Change = abs(Ct - Ct-1)
    VHF = (HCP - LCP) / RollingSum[length] of Change

Args:
    source (pd.Series): Series of prices (usually close).
    length (int): The period length. Default: 28
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
