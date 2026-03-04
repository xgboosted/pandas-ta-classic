# -*- coding: utf-8 -*-
# Moving Average (MA)
from typing import Any, Optional
from pandas import Series

from .dema import dema
from .ema import ema
from .fwma import fwma
from .hma import hma
from .linreg import linreg
from .midpoint import midpoint
from .pwma import pwma
from .rma import rma
from .sinwma import sinwma
from .sma import sma
from .swma import swma
from .t3 import t3
from .tema import tema
from .trima import trima
from .vidya import vidya
from .wma import wma
from .zlma import zlma

_MA_DISPATCH = {
    "dema": dema,
    "ema": ema,
    "fwma": fwma,
    "hma": hma,
    "linreg": linreg,
    "midpoint": midpoint,
    "pwma": pwma,
    "rma": rma,
    "sinwma": sinwma,
    "sma": sma,
    "swma": swma,
    "t3": t3,
    "tema": tema,
    "trima": trima,
    "vidya": vidya,
    "wma": wma,
    "zlma": zlma,
}


def ma(
    name: Optional[str] = None, source: Optional[Series] = None, **kwargs: Any
) -> Optional[Series]:
    """Simple MA Utility for easier MA selection

    Available MAs:
        dema, ema, fwma, hma, linreg, midpoint, pwma, rma,
        sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma

    Examples:
        ema8 = ta.ma("ema", df.close, length=8)
        sma50 = ta.ma("sma", df.close, length=50)
        pwma10 = ta.ma("pwma", df.close, length=10, asc=False)

    Args:
        name (str): One of the Available MAs. Default: "ema"
        source (pd.Series): The 'source' Series.

    Kwargs:
        Any additional kwargs the MA may require.

    Returns:
        pd.Series: New feature generated.
    """
    if name is None and source is None:
        return list(_MA_DISPATCH)

    name = (
        name.lower()
        if isinstance(name, str) and name.lower() in _MA_DISPATCH
        else "ema"
    )
    return _MA_DISPATCH[name](source, **kwargs)
