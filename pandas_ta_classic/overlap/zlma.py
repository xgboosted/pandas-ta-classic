# Zero Lag Moving Average (ZLMA)
from typing import Any, Optional
from pandas import Series
from .dema import dema
from .ema import ema
from .hma import hma
from .linreg import linreg
from .rma import rma
from .sma import sma
from .swma import swma
from .t3 import t3
from .tema import tema
from .trima import trima
from .vidya import vidya
from .wma import wma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series

# Dispatch table: mamode string → MA function.
# "ema" is the catch-all default so it is looked up via dict.get(name, ema).
_ZLMA_DISPATCH = {
    "dema": dema,
    "ema": ema,
    "hma": hma,
    "linreg": linreg,
    "rma": rma,
    "sma": sma,
    "swma": swma,
    "t3": t3,
    "tema": tema,
    "trima": trima,
    "vidya": vidya,
    "wma": wma,
}


def zlma(
    close: Series,
    length: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Zero Lag Moving Average (ZLMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    lag = int(0.5 * (length - 1))
    close_ = 2 * close - close.shift(lag)
    ma_fn = _ZLMA_DISPATCH.get(mamode, ema)
    zlma = ma_fn(close_, length=length, **kwargs)

    if zlma is None:
        return None

    # Offset
    zlma = apply_offset(zlma, offset)

    zlma = apply_fill(zlma, **kwargs)

    # Name & Category
    zlma.name = f"ZL_{zlma.name}"
    zlma.category = "overlap"

    return zlma


zlma.__doc__ = """Zero Lag Moving Average (ZLMA)

The Zero Lag Moving Average attempts to eliminate the lag associated
with moving averages.  This is an adaption created by John Ehler and Ric Way.

Sources:
    https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average

Calculation:
    Default Inputs:
        length=10, mamode=EMA
    EMA = Exponential Moving Average
    lag = int(0.5 * (length - 1))

    SOURCE = 2 * close - close.shift(lag)
    ZLMA = MA(kind=mamode, SOURCE, length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    mamode (str): Options: 'ema', 'hma', 'sma', 'wma'. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
