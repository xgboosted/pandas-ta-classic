# Hilbert Transform - SineWave (HT_SINE)
from typing import Any, Optional

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_offset,
    verify_series,
)


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
    mode_tal = _get_tal_mode(talib)

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

    return _build_dataframe(
        {"HT_SINE": sine, "HT_LEADSINE": lead_sine},
        "HT_SINE",
        "cycles",
        offset,
        **kwargs,
    )


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
