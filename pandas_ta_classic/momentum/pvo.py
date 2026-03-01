# -*- coding: utf-8 -*-
# Percentage Volume Oscillator (PVO)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


def pvo(
    volume: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Percentage Volume Oscillator (PVO)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    if slow < fast:
        fast, slow = slow, fast
    volume = verify_series(volume, max(fast, slow, signal))
    offset = get_offset(offset)

    if volume is None:
        return None

    # Calculate Result
    fastma = ema(volume, length=fast)
    slowma = ema(volume, length=slow)
    pvo = scalar * (fastma - slowma)
    pvo /= slowma

    signalma = ema(pvo, length=signal)
    histogram = pvo - signalma

    # Offset + Name + Category + DataFrame
    _props = f"_{fast}_{slow}_{signal}"
    return _build_dataframe(
        {f"PVO{_props}": pvo, f"PVOh{_props}": histogram, f"PVOs{_props}": signalma},
        f"PVO{_props}",
        "momentum",
        offset,
        **kwargs,
    )


pvo.__doc__ = """Percentage Volume Oscillator (PVO)

Percentage Volume Oscillator is a Momentum Oscillator for Volume.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=PVO.htm

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average

    PVO = (EMA(volume, fast) - EMA(volume, slow)) / EMA(volume, slow)
    Signal = EMA(PVO, signal)
    Histogram = PVO - Signal

Args:
    volume (pd.Series): Series of 'volume's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    signal (int): The signal period. Default: 9
    scalar (float): How much to magnify. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: pvo, histogram, signal columns.
"""
