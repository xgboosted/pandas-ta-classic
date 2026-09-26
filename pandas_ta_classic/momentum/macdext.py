# MACD Extended (MACDEXT)
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input

# TA-Lib MA type integer → string kind for the native path. KAMA (6) and
# MAMA (7) have no native equivalent here; they need TA-Lib.
_MATYPE_TO_KIND = {
    0: "sma",
    1: "ema",
    2: "wma",
    3: "dema",
    4: "tema",
    5: "trima",
    8: "t3",
}


@nan_on_short_input
def macdext(
    close: Series,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    fastmatype: int | None = None,
    slowmatype: int | None = None,
    signalmatype: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: MACD Extended (MACDEXT)

    MACD with independently controllable MA types for fast, slow, and signal
    lines. MA type integers follow the TA-Lib convention:
    0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3.
    """
    # Validate Arguments
    fast = _pos_int(fast, 12, "fast")
    slow = _pos_int(slow, 26, "slow")
    signal = _pos_int(signal, 9, "signal")
    fastmatype = _pos_int(fastmatype, 1, "fastmatype", gt=None, ge=0, lt=9)  # TA-Lib MA_Type 0..8
    slowmatype = _pos_int(slowmatype, 1, "slowmatype", gt=None, ge=0, lt=9)  # TA-Lib MA_Type 0..8
    signalmatype = _pos_int(signalmatype, 1, "signalmatype", gt=None, ge=0, lt=9)  # TA-Lib MA_Type 0..8
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, slow + signal)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
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

        for label, matype in (("fastmatype", fastmatype), ("slowmatype", slowmatype), ("signalmatype", signalmatype)):
            if matype not in _MATYPE_TO_KIND:
                # It used to compute an EMA instead, with only a warning.
                raise ValueError(f"macdext() {label}={matype} (KAMA/MAMA) needs TA-Lib: pass talib=True with TA-Lib installed")
        fast_kind = _MATYPE_TO_KIND[fastmatype]
        slow_kind = _MATYPE_TO_KIND[slowmatype]
        signal_kind = _MATYPE_TO_KIND[signalmatype]

        fast_ma = ma(fast_kind, close, length=fast)
        slow_ma = ma(slow_kind, close, length=slow)
        macd_line = fast_ma - slow_ma
        signal_line = ma(signal_kind, macd_line, length=signal)
        histogram = macd_line - signal_line

    macd_series = Series(np.array(macd_line, dtype=float), index=close.index)
    signal_series = Series(np.array(signal_line, dtype=float), index=close.index)
    histogram_series = Series(np.array(histogram, dtype=float), index=close.index)

    # Offset
    macd_series, signal_series, histogram_series = apply_offset([macd_series, signal_series, histogram_series], offset)

    # Handle fills
    macd_series, signal_series, histogram_series = apply_fill([macd_series, signal_series, histogram_series], **kwargs)

    # Name and Categorize
    _params = f"_{fast}_{slow}_{signal}"
    macd_series.name = f"MACDEXT{_params}"
    signal_series.name = f"MACDEXTs{_params}"
    histogram_series.name = f"MACDEXTh{_params}"
    macd_series.category = signal_series.category = histogram_series.category = "momentum"

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
    0=SMA, 1=EMA (default), 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA*, 7=MAMA*, 8=T3
    (* native fallback uses EMA; UserWarning emitted)

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
        Native fallback supports: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA,
        5=TRIMA, 8=T3. Types 6=KAMA and 7=MAMA use EMA as a fallback and
        emit a UserWarning. Use talib=True for correct KAMA/MAMA behaviour.
    talib (bool): Use TA-Lib if available. Default: False.
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
