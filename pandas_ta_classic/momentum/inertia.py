# -*- coding: utf-8 -*-
# Inertia (INERTIA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.volatility import rvi
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def inertia(
    close: Optional[Series] = None,
    high: Optional[Series] = None,
    low: Optional[Series] = None,
    length: Optional[int] = None,
    rvi_length: Optional[int] = None,
    scalar: Optional[float] = None,
    refined: Optional[bool] = None,
    thirds: Optional[bool] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Inertia (INERTIA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 20
    rvi_length = int(rvi_length) if rvi_length and rvi_length > 0 else 14
    scalar = float(scalar) if scalar and scalar > 0 else 100
    refined = False if refined is None else True
    thirds = False if thirds is None else True
    mamode = mamode if isinstance(mamode, str) else "ema"
    _length = max(length, rvi_length)
    close = verify_series(close, _length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    if refined or thirds:
        high = verify_series(high, _length)
        low = verify_series(low, _length)
        if high is None or low is None:
            return None

    # Calculate Result
    if refined:
        _mode, rvi_ = "r", rvi(
            close,
            high=high,
            low=low,
            length=rvi_length,
            scalar=scalar,
            refined=refined,
            mamode=mamode,
        )
    elif thirds:
        _mode, rvi_ = "t", rvi(
            close,
            high=high,
            low=low,
            length=rvi_length,
            scalar=scalar,
            thirds=thirds,
            mamode=mamode,
        )
    else:
        _mode, rvi_ = "", rvi(close, length=rvi_length, scalar=scalar, mamode=mamode)

    if rvi_ is None:
        return None
    inertia = linreg(rvi_, length=length)
    if inertia is None:
        return None

    # Offset
    inertia = apply_offset(inertia, offset)

    inertia = apply_fill(inertia, **kwargs)

    # Name & Category
    _props = f"_{length}_{rvi_length}"
    inertia.name = f"INERTIA{_mode}{_props}"
    inertia.category = "momentum"

    return inertia


inertia.__doc__ = """Inertia (INERTIA)

Inertia was developed by Donald Dorsey and was introduced his article
in September, 1995. It is the Relative Vigor Index smoothed by the Least
Squares Moving Average. Postive Inertia when values are greater than 50,
Negative Inertia otherwise.

Sources:
    https://www.investopedia.com/terms/r/relative_vigor_index.asp

Calculation:
    Default Inputs:
        length=14, ma_length=20
    LSQRMA = Least Squares Moving Average

    INERTIA = LSQRMA(RVI(length), ma_length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    rvi_length (int): RVI period. Default: 14
    refined (bool): Use 'refined' calculation. Default: False
    thirds (bool): Use 'thirds' calculation. Default: False
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
