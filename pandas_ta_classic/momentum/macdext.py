# -*- coding: utf-8 -*-
# MACD Extended (MACDEXT)
import warnings
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series

# TA-Lib MA type integer → string kind for native fallback
_MATYPE_TO_KIND = {
    0: "sma",
    1: "ema",
    2: "wma",
    3: "dema",
    4: "tema",
    5: "trima",
    6: "kama",
    7: "mama",
    8: "t3",
}


def macdext(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    fastmatype: Optional[int] = None,
    slowmatype: Optional[int] = None,
    signalmatype: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: MACD Extended (MACDEXT)

    MACD with independently controllable MA types for fast, slow, and signal
    lines. MA type integers follow the TA-Lib convention:
    0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3.
    """
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    fastmatype = int(fastmatype) if fastmatype is not None and fastmatype >= 0 else 1
    slowmatype = int(slowmatype) if slowmatype is not None and slowmatype >= 0 else 1
    signalmatype = (
        int(signalmatype) if signalmatype is not None and signalmatype >= 0 else 1
    )
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, slow + signal)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MACDEXT as _MACDEXT

        macd_line, signal_line, histogram = _MACDEXT(
            close,
            fastperiod=fast,
            slowperiod=slow,
            fastmatype=fastmatype,
            slowmatype=slowmatype,
            signalperiod=signal,
            signalmatype=signalmatype,
        )
    else:
        from pandas_ta_classic.overlap.ma import ma

        # matypes 6 (KAMA) and 7 (MAMA) are not implemented natively;
        # they silently fall back to EMA inside ma(). Warn the caller.
        _unsupported = {6: "kama", 7: "mama"}
        for _mt, _name in _unsupported.items():
            if _mt in (fastmatype, slowmatype, signalmatype):
                warnings.warn(
                    f"MACDEXT native fallback does not support matype={_mt} ({_name}); "
                    f"EMA will be used instead. Pass talib=True (with TA-Lib installed) "
                    f"to get the correct {_name.upper()} behaviour.",
                    UserWarning,
                    stacklevel=2,
                )

        fast_kind = _MATYPE_TO_KIND.get(fastmatype, "ema")
        slow_kind = _MATYPE_TO_KIND.get(slowmatype, "ema")
        signal_kind = _MATYPE_TO_KIND.get(signalmatype, "ema")

        fast_ma = ma(fast_kind, close, length=fast)
        slow_ma = ma(slow_kind, close, length=slow)
        macd_line = fast_ma - slow_ma
        signal_line = ma(signal_kind, macd_line, length=signal)
        histogram = macd_line - signal_line

    macd_series = Series(np.array(macd_line, dtype=float), index=close.index)
    signal_series = Series(np.array(signal_line, dtype=float), index=close.index)
    histogram_series = Series(np.array(histogram, dtype=float), index=close.index)

    # Offset
    macd_series, signal_series, histogram_series = apply_offset(
        [macd_series, signal_series, histogram_series], offset
    )

    # Handle fills
    macd_series, signal_series, histogram_series = apply_fill(
        [macd_series, signal_series, histogram_series], **kwargs
    )

    # Name and Categorize
    _params = f"_{fast}_{slow}_{signal}"
    macd_series.name = f"MACDEXT{_params}"
    signal_series.name = f"MACDEXTs{_params}"
    histogram_series.name = f"MACDEXTh{_params}"
    macd_series.category = signal_series.category = histogram_series.category = (
        "momentum"
    )

    df = DataFrame(
        {
            macd_series.name: macd_series,
            signal_series.name: signal_series,
            histogram_series.name: histogram_series,
        },
        index=close.index,
    )
    df.name = f"MACDEXT{_params}"
    df.category = "momentum"
    return df


macdext.__doc__ = """
MACD Extended (MACDEXT)

Like MACD but each of the three moving averages (fast, slow, signal) can use
a different MA type, following the TA-Lib convention.

MA type integers:
    0=SMA, 1=EMA (default), 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3

Sources:
    TA-Lib: https://ta-lib.org/functions/

Args:
    close (pd.Series): Close price series.
    fast (int): Fast period. Default: 12.
    slow (int): Slow period. Default: 26.
    signal (int): Signal period. Default: 9.
    fastmatype (int): MA type for fast line. Default: 1 (EMA).
    slowmatype (int): MA type for slow line. Default: 1 (EMA).
    signalmatype (int): MA type for signal line. Default: 1 (EMA).
    talib (bool): Use TA-Lib if available. Default: True.
    offset (int): Number of periods to offset. Default: 0.

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Columns MACDEXT_{f}_{s}_{sig}, MACDEXTs_{f}_{s}_{sig},
                  MACDEXTh_{f}_{s}_{sig}.

Example:
    df[['MACDEXT_12_26_9', 'MACDEXTs_12_26_9', 'MACDEXTh_12_26_9']] = df.ta.macdext()
    # SMA-based MACD:
    df.ta.macdext(fastmatype=0, slowmatype=0, signalmatype=0)
"""
