# Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import _get_tal_mode, _finalize, get_offset, verify_series


def ht_dcphase(
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Dominant Cycle Phase"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import HT_DCPHASE as taHT

        result = Series(taHT(close), index=close.index)
    else:
        ht = hilbert_result(close, ht_start=37)
        result = Series(ht["dc_phase"], index=close.index)

    return _finalize(result, offset, "HT_DCPHASE", "cycles", **kwargs)


ht_dcphase.__doc__ = """Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)

The Dominant Cycle Phase uses the Hilbert Transform to estimate the
phase of the dominant cycle.

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
