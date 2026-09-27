# MACD with Fixed Periods (MACDFIX)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.momentum.macd import macd
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


@nan_on_short_input
def macdfix(
    close: Series,
    signal: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: MACD with Fixed Periods (MACDFIX)

    MACD with fixed fast=12, slow=26, variable signal period.
    TA-Lib name: MACDFIX.
    """
    signal = _pos_int(signal, 9, "signal")
    close = verify_series(close, 26 + signal)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Remove fast/slow from kwargs to avoid duplicate keyword argument errors
    kwargs.pop("fast", None)
    kwargs.pop("slow", None)

    # The offset and the fill are applied once, below, to whichever branch ran.
    # macd() would otherwise apply them a second time on the native path.
    fill_kwargs = {key: kwargs.pop(key) for key in ("fillna", "fill_method") if key in kwargs}

    # Read without popping: the native branch forwards it to macd(), which owns
    # the signal columns and the options that go with them.
    signal_indicators = _bool_param(kwargs.get("signal_indicators", None), False, "signal_indicators")

    # TA-Lib's MACDFIX returns the three lines only; run natively instead of
    # dropping the signal columns it cannot produce
    if Imports["talib"] and mode_talib and not signal_indicators:
        from talib import MACDFIX as _MACDFIX

        macd_line, signal_line, hist = _MACDFIX(close, signalperiod=signal)
        data = {
            f"MACDFIX_{signal}_{signal}": macd_line,
            f"MACDFIXh_{signal}_{signal}": hist,
            f"MACDFIXs_{signal}_{signal}": signal_line,
        }
        result = DataFrame(data, index=close.index)
    else:
        result = macd(close, fast=12, slow=26, signal=signal, talib=False, **kwargs)
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
    result = apply_fill(result, **fill_kwargs)

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
    talib (bool): Use TA-Lib if available. Default: False.
    offset (int): Number of periods to offset the result. Default: 0.

Kwargs:
    signal_indicators (bool): When True, the signal columns macd() produces
        are appended; the TA-Lib path cannot produce them, so talib=True is
        ignored while it is set. The xa, xb, cross_values, xserie, xserie_a,
        xserie_b and cross_series options are forwarded to macd(). Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: DataFrame with MACDFIX line, histogram, signal columns.
"""
