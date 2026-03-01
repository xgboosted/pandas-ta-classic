# -*- coding: utf-8 -*-
# Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import _get_tal_mode, _finalize, get_offset, verify_series


def ht_trendline(
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Instantaneous Trendline"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import HT_TRENDLINE as taHT

        result = Series(taHT(close), index=close.index)
    else:
        ht = hilbert_result(close, ht_start=37)
        result = Series(ht["trendline"], index=close.index)

    return _finalize(result, offset, "HT_TRENDLINE", "overlap", **kwargs)


ht_trendline.__doc__ = """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE)

The Instantaneous Trendline uses the Hilbert Transform dominant cycle
period to compute a smoothed trendline that adapts to the current
market cycle.

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
    pd.Series: New feature generated.
"""
