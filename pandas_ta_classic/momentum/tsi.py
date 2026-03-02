# True Strength Index (TSI)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_drift,
    get_offset,
    verify_series,
)


def tsi(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: True Strength Index (TSI)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 13
    slow = int(slow) if slow and slow > 0 else 25
    signal = int(signal) if signal and signal > 0 else 13
    # if slow < fast:
    #     fast, slow = slow, fast
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, max(fast, slow))
    drift = get_drift(drift)
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "ema"
    if "length" in kwargs:
        kwargs.pop("length")

    if close is None:
        return None

    # Calculate Result
    diff = close.diff(drift)
    slow_ema = ema(close=diff, length=slow, **kwargs)
    fast_slow_ema = ema(close=slow_ema, length=fast, **kwargs)

    abs_diff = diff.abs()
    abs_slow_ema = ema(close=abs_diff, length=slow, **kwargs)
    abs_fast_slow_ema = ema(close=abs_slow_ema, length=fast, **kwargs)

    tsi = scalar * fast_slow_ema / abs_fast_slow_ema
    tsi_signal = ma(mamode, tsi, length=signal)
    if tsi_signal is None:
        return None

    # Offset + Name + Category + DataFrame
    _props = f"_{fast}_{slow}_{signal}"
    return _build_dataframe(
        {f"TSI{_props}": tsi, f"TSIs{_props}": tsi_signal},
        f"TSI{_props}",
        "momentum",
        offset,
        **kwargs,
    )


tsi.__doc__ = """True Strength Index (TSI)

The True Strength Index is a momentum indicator used to identify short-term
swings while in the direction of the trend as well as determining overbought
and oversold conditions.

Sources:
    https://www.investopedia.com/terms/t/tsi.asp

Calculation:
    Default Inputs:
        fast=13, slow=25, signal=13, scalar=100, drift=1
    EMA = Exponential Moving Average
    diff = close.diff(drift)

    slow_ema = EMA(diff, slow)
    fast_slow_ema = EMA(slow_ema, slow)

    abs_diff_slow_ema = absolute_diff_ema = EMA(ABS(diff), slow)
    abema = abs_diff_fast_slow_ema = EMA(abs_diff_slow_ema, fast)

    TSI = scalar * fast_slow_ema / abema
    Signal = EMA(TSI, signal)

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 13
    slow (int): The long period. Default: 25
    signal (int): The signal period. Default: 13
    scalar (float): How much to magnify. Default: 100
    mamode (str): Moving Average of TSI Signal Line.
        See ```help(ta.ma)```. Default: 'ema'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: tsi, signal.
"""
