# Klinger Volume Oscillator (KVO)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.hlc3 import hlc3
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    signed_series,
    verify_series,
)
from pandas_ta_classic.utils._core import _pos_int, _str_param, nan_on_short_input, skip_leading_nan


@nan_on_short_input
@skip_leading_nan("high", "low", "close", "volume")
def kvo(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Klinger Volume Oscillator (KVO)"""
    # Validate arguments
    fast = _pos_int(fast, 34, "fast")
    slow = _pos_int(slow, 55, "slow")
    signal = _pos_int(signal, 13, "signal")
    mamode = _str_param(mamode, "ema", "mamode")
    _length = max(fast, slow, signal)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)
    # A strategy-wide drift (df.ta.strategy(..., drift=N)) has no meaning here: the
    # parameter was removed in 0.9.0. Drop it rather than forward it to apply_fill.
    kwargs.pop("drift", None)

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
    kvo, kvo_signal = apply_offset([kvo, kvo_signal], offset)

    kvo, kvo_signal = apply_fill([kvo, kvo_signal], **kwargs)

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
        fast=34, slow=55, signal=13
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
