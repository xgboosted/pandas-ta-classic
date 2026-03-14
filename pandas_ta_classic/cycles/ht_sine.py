# -*- coding: utf-8 -*-
# Hilbert Transform - SineWave (HT_SINE)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import get_offset, verify_series


def ht_sine(
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Hilbert Transform - SineWave"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import HT_SINE as taHT

        sine_arr, lead_arr = taHT(close)
        sine = Series(sine_arr, index=close.index)
        lead_sine = Series(lead_arr, index=close.index)
    else:
        ht = hilbert_result(close, ht_start=37)
        sine = Series(ht["sine"], index=close.index)
        lead_sine = Series(ht["lead_sine"], index=close.index)

    # Offset
    if offset != 0:
        sine = sine.shift(offset)
        lead_sine = lead_sine.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        sine.fillna(kwargs["fillna"], inplace=True)
        lead_sine.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            sine.ffill(inplace=True)
            lead_sine.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            sine.bfill(inplace=True)
            lead_sine.bfill(inplace=True)

    # Name and Categorize it
    sine.name = "HT_SINE"
    lead_sine.name = "HT_LEADSINE"

    data = {sine.name: sine, lead_sine.name: lead_sine}
    df = DataFrame(data)
    df.name = "HT_SINE"
    df.category = "cycles"

    return df


ht_sine.__doc__ = """Hilbert Transform - SineWave (HT_SINE)

Returns the Sine and LeadSine of the dominant cycle phase.  Crossovers
of the Sine and LeadSine can be used as cycle-mode trading signals.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: sine and leadsine columns.
"""
