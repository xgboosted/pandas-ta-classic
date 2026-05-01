# -*- coding: utf-8 -*-
# MACD with Fixed Periods (MACDFIX)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.momentum.macd import macd
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def macdfix(
    close: Series,
    signal: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: MACD with Fixed Periods (MACDFIX)

    MACD with fixed fast=12, slow=26, variable signal period.
    TA-Lib name: MACDFIX.
    """
    signal = int(signal) if signal and signal > 0 else 9
    close = verify_series(close, 26 + signal)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Remove fast/slow from kwargs to avoid duplicate keyword argument errors
    kwargs.pop("fast", None)
    kwargs.pop("slow", None)

    if Imports["talib"] and mode_tal:
        from talib import MACDFIX as _MACDFIX

        macd_line, signal_line, hist = _MACDFIX(close, signalperiod=signal)
        data = {
            f"MACDFIX_{signal}_{signal}": macd_line,
            f"MACDFIXh_{signal}_{signal}": hist,
            f"MACDFIXs_{signal}_{signal}": signal_line,
        }
        result = DataFrame(data, index=close.index)
    else:
        result = macd(
            close, fast=12, slow=26, signal=signal, talib=False, offset=offset, **kwargs
        )
        if result is None:
            return None

        # Rename columns to MACDFIX convention
        cols = result.columns.tolist()
        new_cols = {}
        for col in cols:
            new_col = (
                col.replace(f"MACD_{12}_{26}_{signal}", f"MACDFIX_{signal}_{signal}")
                .replace(f"MACDh_{12}_{26}_{signal}", f"MACDFIXh_{signal}_{signal}")
                .replace(f"MACDs_{12}_{26}_{signal}", f"MACDFIXs_{signal}_{signal}")
            )
            new_cols[col] = new_col
        result = result.rename(columns=new_cols)

    # Offset
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)

    result.name = f"MACDFIX_{signal}"
    result.category = "momentum"
    return result


macdfix.__doc__ = """MACD with Fixed Periods (MACDFIX)

MACD calculated with fixed periods: fast=12, slow=26, configurable signal.
Uses TA-Lib MACDFIX when available (which uses a different EMA initialization
than MACD), otherwise falls back to MACD(12, 26, signal) with native EMA.

TA-Lib name: MACDFIX.

Args:
    close (pd.Series): Series of 'close' prices.
    signal (int): Signal period. Default: 9.
    talib (bool): Use TA-Lib if available. Default: True.
    offset (int): Number of periods to offset the result. Default: 0.

Returns:
    pd.DataFrame: DataFrame with MACDFIX line, histogram, signal columns.
"""


macdfix.__doc__ = """MACD with Fixed Periods (MACDFIX)

MACD using fixed fast=12 and slow=26 with a variable signal period.
Equivalent to ta.macd(close, fast=12, slow=26, signal=signalperiod).

TA-Lib name: MACDFIX.

Args:
    close (pd.Series): Series of 'close' prices
    signal (int): Signal period. Default: 9
    talib (bool): Use TA-Lib C library if installed. Default: True
    offset (int): Periods to offset. Default: 0

Returns:
    pd.DataFrame: macdfix, histogram, signal columns.
"""
