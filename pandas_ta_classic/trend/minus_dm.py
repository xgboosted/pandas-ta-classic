# -*- coding: utf-8 -*-
# Minus Directional Movement (MINUS_DM)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
    zero,
)


def minus_dm(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Minus Directional Movement (-DM, MINUS_DM)

    Raw Wilder-smoothed negative directional movement.
    TA-Lib name: MINUS_DM.
    """
    length = int(length) if length and length > 0 else 14
    high = verify_series(high, length)
    low = verify_series(low, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None:
        return None

    if Imports["talib"] and mode_tal:
        from talib import MINUS_DM as _MINUS_DM

        result = _MINUS_DM(high, low, timeperiod=length)
    else:
        from pandas_ta_classic.overlap.ma import ma

        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        neg_ = ((dn > up) & (dn > 0)) * dn
        neg_ = neg_.apply(zero)
        result = ma("rma", neg_, length=length)
        if result is None:
            return None
        result = result * length  # Wilder's raw DM

    # Offset
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)

    result.name = f"MINUS_DM_{length}"
    result.category = "trend"
    return result


minus_dm.__doc__ = """Minus Directional Movement (-DM, MINUS_DM)

Raw Wilder-smoothed negative directional movement before conversion to DI-.

TA-Lib name: MINUS_DM.

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    length (int): Lookback period. Default: 14
    talib (bool): Use TA-Lib C library if installed. Default: True
    drift (int): Difference period. Default: 1
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
