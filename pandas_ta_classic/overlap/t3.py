# -*- coding: utf-8 -*-
# T3 (T3)
from typing import Any, Optional
from pandas import Series
from .ema import ema, _ema_chain
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def t3(
    close: Series,
    length: Optional[int] = None,
    a: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: T3"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    a = float(a) if a and a > 0 and a < 1 else 0.7
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import T3

        t3 = T3(close, length, a)
    else:
        c1 = -a * a**2
        c2 = 3 * a**2 + 3 * a**3
        c3 = -6 * a**2 - 3 * a - 3 * a**3
        c4 = a**3 + 3 * a**2 + 3 * a + 1

        emas = _ema_chain(close, length, 6, **kwargs)
        if emas is None or len(emas) < 6:
            return None
        _, _, e3, e4, e5, e6 = emas
        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # Offset
    t3 = apply_offset(t3, offset)

    t3 = apply_fill(t3, **kwargs)

    # Name & Category
    t3.name = f"T3_{length}_{a}"
    t3.category = "overlap"

    return t3


t3.__doc__ = """Tim Tillson's T3 Moving Average (T3)

Tim Tillson's T3 Moving Average is considered a smoother and more responsive
moving average relative to other moving averages.

Sources:
    http://www.binarytribune.com/forex-trading-indicators/t3-moving-average-indicator/

Calculation:
    Default Inputs:
        length=10, a=0.7
    c1 = -a^3
    c2 = 3a^2 + 3a^3 = 3a^2 * (1 + a)
    c3 = -6a^2 - 3a - 3a^3
    c4 = a^3 + 3a^2 + 3a + 1

    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    ema4 = EMA(ema3, length)
    ema5 = EMA(ema4, length)
    ema6 = EMA(ema5, length)
    T3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    a (float): 0 < a < 1. Default: 0.7
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
