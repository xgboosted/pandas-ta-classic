# -*- coding: utf-8 -*-
# Triple Exponential Moving Average (TEMA)
from typing import Any, Optional
from pandas import Series
from .ema import _ema_chain
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def tema(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Triple Exponential Moving Average (TEMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import TEMA

        tema = TEMA(close, length)
    else:
        emas = _ema_chain(close, length, 3, **kwargs)
        if emas is None or len(emas) < 3:
            return None
        ema1, ema2, ema3 = emas[0], emas[1], emas[2]
        tema = 3 * (ema1 - ema2) + ema3

    # Offset
    tema = apply_offset(tema, offset)

    tema = apply_fill(tema, **kwargs)

    # Name & Category
    tema.name = f"TEMA_{length}"
    tema.category = "overlap"

    return tema


tema.__doc__ = """Triple Exponential Moving Average (TEMA)

A less laggy Exponential Moving Average.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TEMA = 3 * (ema1 - ema2) + ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
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
