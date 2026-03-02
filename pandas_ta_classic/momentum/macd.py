# -*- coding: utf-8 -*-
# Moving Average Convergence Divergence (MACD)
from typing import Any, Optional
from pandas import concat, DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _swap_fast_slow,
    _build_dataframe,
    get_offset,
    signals,
    verify_series,
)


def macd(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    fast, slow = _swap_fast_slow(fast, slow)
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    as_mode = kwargs.setdefault("asmode", False)

    # Calculate Result
    # TA-Lib MACD only supports EMA; use native path for other mamodes
    if Imports["talib"] and mode_tal and mamode == "ema":
        from talib import MACD

        macd, signalma, histogram = MACD(close, fast, slow, signal)
    else:
        fastma = ma(mamode, close, length=fast)
        slowma = ma(mamode, close, length=slow)

        macd = fastma - slowma
        signalma = ma(mamode, macd.loc[macd.first_valid_index() :,], length=signal)
        histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ma(mamode, macd.loc[macd.first_valid_index() :,], length=signal)
        histogram = macd - signalma

    # Offset + Name + Category + DataFrame
    _asmode = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    df = _build_dataframe(
        {
            f"MACD{_asmode}{_props}": macd,
            f"MACD{_asmode}h{_props}": histogram,
            f"MACD{_asmode}s{_props}": signalma,
        },
        f"MACD{_asmode}{_props}",
        "momentum",
        offset,
        **kwargs,
    )

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = concat(
            [
                df,
                signals(
                    indicator=histogram,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", True),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
                signals(
                    indicator=macd,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return df


macd.__doc__ = """Moving Average Convergence Divergence (MACD)

The MACD is a popular indicator to that is used to identify a security's trend.
While APO and MACD are the same calculation, MACD also returns two more series
called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
the difference of MACD and Signal.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
    AS Mode: https://tr.tradingview.com/script/YFlKXHnP/

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

    if asmode:
        MACD = MACD - Signal
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    signal (int): The signal period. Default: 9
    mamode (str): See ``help(ta.ma)``. Default: 'ema'
        When set to a value other than 'ema', the native calculation is used
        regardless of the talib flag (MACDEXT behaviour).
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    asmode (value, optional): When True, enables AS version of MACD.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns.
"""
