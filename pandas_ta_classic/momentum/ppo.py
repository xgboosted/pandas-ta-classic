# -*- coding: utf-8 -*-
# Percentage Price Oscillator (PPO)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _swap_fast_slow,
    _build_dataframe,
    get_offset,
    tal_ma,
    verify_series,
)


def ppo(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Percentage Price Oscillator (PPO)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    mamode = mamode if isinstance(mamode, str) else "sma"
    fast, slow = _swap_fast_slow(fast, slow)
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import PPO

        ppo = PPO(close, fast, slow, tal_ma(mamode))
    else:
        fastma = ma(mamode, close, length=fast)
        slowma = ma(mamode, close, length=slow)
        if fastma is None or slowma is None:
            return None
        ppo = scalar * (fastma - slowma)
        ppo /= slowma

    signalma = ma("ema", ppo, length=signal)
    if signalma is None:
        return None
    histogram = ppo - signalma

    # Offset + Name + Category + DataFrame
    _props = f"_{fast}_{slow}_{signal}"
    return _build_dataframe(
        {f"PPO{_props}": ppo, f"PPOh{_props}": histogram, f"PPOs{_props}": signalma},
        f"PPO{_props}",
        "momentum",
        offset,
        **kwargs,
    )


ppo.__doc__ = """Percentage Price Oscillator (PPO)

The Percentage Price Oscillator is similar to MACD in measuring momentum.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Calculation:
    Default Inputs:
        fast=12, slow=26
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    fast_sma = SMA(close, fast)
    slow_sma = SMA(close, slow)
    PPO = 100 * (fast_sma - slow_sma) / slow_sma
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal

Args:
    close(pandas.Series): Series of 'close's
    fast(int): The short period. Default: 12
    slow(int): The long period. Default: 26
    signal(int): The signal period. Default: 9
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset(int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ppo, histogram, signal columns
"""
