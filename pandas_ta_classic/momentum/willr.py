# Williams %R (WILLR)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


@nan_on_short_input
def willr(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: William's Percent R (WILLR)"""
    # Validate arguments
    length = _pos_int(length, 14, "length")
    min_periods = _pos_int(kwargs.get("min_periods"), length, "min_periods", gt=None, ge=0)
    _length = max(length, min_periods)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import WILLR

        willr = WILLR(high, low, close, length)
    else:
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        highest_high = high.rolling(length, min_periods=min_periods).max()

        # A window where every bar has the same high and low divides 0/0 to
        # NaN, and the NaN then spreads over the next length-1 windows: a flat
        # block of b bars costs max(0, b - (length - 1)) extra NaN, and a fully
        # flat series is all NaN.
        #
        # The window reads 0.0, TA-Lib's marker for a degenerate window. An
        # epsilon denominator would read -100 here instead, because of the
        # affine term: 100 * (0/eps - 1). That is a real %R value -- bar 72 of
        # a series with a flat block from bar 60 genuinely reads -100 -- so it
        # cannot double as the marker. See tests/test_degenerate_input.py.
        window_range = highest_high - lowest_low
        willr = 100 * ((close - lowest_low) / window_range.where(window_range != 0) - 1)
        willr = willr.mask(window_range == 0, 0.0)

    # Offset
    willr = apply_offset(willr, offset)

    willr = apply_fill(willr, **kwargs)

    # Name and Categorize it
    willr.name = f"WILLR_{length}"
    willr.category = "momentum"

    return willr


willr.__doc__ = """William's Percent R (WILLR)

William's Percent R is a momentum oscillator similar to the RSI that
attempts to identify overbought and oversold conditions.

Sources:
    https://www.tradingview.com/wiki/Williams_%25R_(%25R)

Calculation:
    Default Inputs:
        length=20
    LL = low.rolling(length).min()
    HH = high.rolling(length).max()

    WILLR = 100 * ((close - LL) / (HH - LL) - 1)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
