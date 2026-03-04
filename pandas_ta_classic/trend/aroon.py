# -*- coding: utf-8 -*-
# Aroon (AROON)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_offset,
    recent_maximum_index,
    recent_minimum_index,
    verify_series,
)


def aroon(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Aroon & Aroon Oscillator"""
    # Validate Arguments
    length = length if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import AROON, AROONOSC

        aroon_down, aroon_up = AROON(high, low, length)
        aroon_osc = AROONOSC(high, low, length)
    else:
        periods_from_hh = high.rolling(length + 1).apply(recent_maximum_index, raw=True)
        periods_from_ll = low.rolling(length + 1).apply(recent_minimum_index, raw=True)

        aroon_up = aroon_down = scalar
        aroon_up *= 1 - (periods_from_hh / length)
        aroon_down *= 1 - (periods_from_ll / length)
        aroon_osc = aroon_up - aroon_down

    # Offset, Name and Categorize it
    return _build_dataframe(
        {
            f"AROOND_{length}": aroon_down,
            f"AROONU_{length}": aroon_up,
            f"AROONOSC_{length}": aroon_osc,
        },
        f"AROON_{length}",
        "trend",
        offset,
        **kwargs,
    )


aroon.__doc__ = """Aroon & Aroon Oscillator (AROON)

Aroon attempts to identify if a security is trending and how strong.

Sources:
    https://www.tradingview.com/wiki/Aroon
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/aroon-ar/

Calculation:
    Default Inputs:
        length=1, scalar=100

    recent_maximum_index(x): return int(np.argmax(x[::-1]))
    recent_minimum_index(x): return int(np.argmin(x[::-1]))

    periods_from_hh = high.rolling(length + 1).apply(recent_maximum_index, raw=True)
    AROON_UP = scalar * (1 - (periods_from_hh / length))

    periods_from_ll = low.rolling(length + 1).apply(recent_minimum_index, raw=True)
    AROON_DN = scalar * (1 - (periods_from_ll / length))

    AROON_OSC = AROON_UP - AROON_DN

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    scalar (float): How much to magnify. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: aroon_up, aroon_down, aroon_osc columns.
"""
