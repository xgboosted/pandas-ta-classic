# Double Exponential Moving Average (DEMA)
from typing import Any, Optional
from pandas import Series
from .ema import _ema_chain
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def dema(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Double Exponential Moving Average (DEMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import DEMA

        dema = DEMA(close, length)
    else:
        emas = _ema_chain(close, length, 2)
        if emas is None or len(emas) < 2:
            return None
        ema1, ema2 = emas[0], emas[1]
        dema = 2 * ema1 - ema2

    # Offset
    dema = apply_offset(dema, offset)
    dema = apply_fill(dema, **kwargs)

    # Name & Category
    dema.name = f"DEMA_{length}"
    dema.category = "overlap"

    return dema


dema.__doc__ = """Double Exponential Moving Average (DEMA)

The Double Exponential Moving Average attempts to a smoother average with less
lag than the normal Exponential Moving Average (EMA).

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)

    DEMA = 2 * ema1 - ema2

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
