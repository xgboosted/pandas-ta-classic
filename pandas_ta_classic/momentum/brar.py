# -*- coding: utf-8 -*-
# BRAR (Bull and Bear Ratio)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def brar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: BRAR (BRAR)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 26
    scalar = float(scalar) if scalar else 100
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    high_open_range = non_zero_range(high, open_)
    open_low_range = non_zero_range(open_, low)

    # Calculate Result
    hcy = non_zero_range(high, close.shift(drift))
    cyl = non_zero_range(close.shift(drift), low)

    hcy[hcy < 0] = 0  # Zero negative values
    cyl[cyl < 0] = 0  # ""

    ar = scalar * high_open_range.rolling(length).sum()
    ar /= open_low_range.rolling(length).sum()

    br = scalar * hcy.rolling(length).sum()
    br /= cyl.rolling(length).sum()

    # Offset + Name + Category + DataFrame
    _props = f"_{length}"
    return _build_dataframe(
        {f"AR{_props}": ar, f"BR{_props}": br},
        f"BRAR{_props}",
        "momentum",
        offset,
        **kwargs,
    )


brar.__doc__ = """BRAR (BRAR)

BR and AR

Sources:
    No internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=26, scalar=100
    SUM = Sum

    HO_Diff = high - open
    OL_Diff = open - low
    HCY = high - close[-1]
    CYL = close[-1] - low
    HCY[HCY < 0] = 0
    CYL[CYL < 0] = 0
    AR = scalar * SUM(HO, length) / SUM(OL, length)
    BR = scalar * SUM(HCY, length) / SUM(CYL, length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 26
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ar, br columns.
"""
