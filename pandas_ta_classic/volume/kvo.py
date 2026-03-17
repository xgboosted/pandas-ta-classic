# -*- coding: utf-8 -*-
# Klinger Volume Oscillator (KVO)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import get_drift, get_offset, signed_series, verify_series


def kvo(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Klinger Volume Oscillator (KVO)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 34
    slow = int(slow) if slow and slow > 0 else 55
    signal = int(signal) if signal and signal > 0 else 13
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    _length = max(fast, slow, signal)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result
    signed_volume = volume * signed_series(hlc3(high, low, close), 1)
    sv = signed_volume.loc[signed_volume.first_valid_index() :,]
    _kvo_fast = ma(mamode, sv, length=fast)
    if _kvo_fast is None:
        return None
    _kvo_slow = ma(mamode, sv, length=slow)
    if _kvo_slow is None:
        return None
    kvo = _kvo_fast - _kvo_slow
    kvo_signal = ma(mamode, kvo.loc[kvo.first_valid_index() :,], length=signal)
    if kvo_signal is None:
        return None

    # Offset
    if offset != 0:
        kvo = kvo.shift(offset)
        kvo_signal = kvo_signal.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        kvo.fillna(kwargs["fillna"], inplace=True)
        kvo_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                kvo.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                kvo.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                kvo_signal.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                kvo_signal.bfill(inplace=True)

    # Name and Categorize it
    _props = f"_{fast}_{slow}_{signal}"
    kvo.name = f"KVO{_props}"
    kvo_signal.name = f"KVOs{_props}"
    kvo.category = kvo_signal.category = "volume"

    # Prepare DataFrame to return
    data = {kvo.name: kvo, kvo_signal.name: kvo_signal}
    df = DataFrame(data)
    df.name = f"KVO{_props}"
    df.category = kvo.category

    return df


kvo.__doc__ = """Klinger Volume Oscillator (KVO)

This indicator was developed by Stephen J. Klinger. It is designed to predict
price reversals in a market by comparing volume to price.

Sources:
    https://www.investopedia.com/terms/k/klingeroscillator.asp
    https://www.daytrading.com/klinger-volume-oscillator

Calculation:
    Default Inputs:
        fast=34, slow=55, signal=13, drift=1
    EMA = Exponential Moving Average

    SV = volume * signed_series(HLC3, 1)
    KVO = EMA(SV, fast) - EMA(SV, slow)
    Signal = EMA(KVO, signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    fast (int): The fast period. Default: 34
    long (int): The long period. Default: 55
    length_sig (int): The signal period. Default: 13
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: KVO and Signal columns.
"""
