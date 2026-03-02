# Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import _get_tal_mode, _finalize, get_offset, verify_series


def ht_trendmode(
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Trend vs Cycle Mode"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import HT_TRENDMODE as taHT

        result = Series(taHT(close), index=close.index)
    else:
        ht = hilbert_result(close, ht_start=37)
        result = Series(ht["trend_mode"], index=close.index)
        # TA-Lib lookback for HT_TRENDMODE is 63; the Hilbert variables
        # have not converged before that.  Blank the warmup zone so the
        # fillna below converts them to 0, matching TA-Lib output.
        result.iloc[:63] = np.nan

    # Convert to nullable Int64 to handle NaN + integer coexistence
    result = result.fillna(-1).astype(int).replace(-1, 0)

    return _finalize(result, offset, "HT_TRENDMODE", "cycles", **kwargs)


ht_trendmode.__doc__ = """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)

Returns 1 when the market is in a trend and 0 when it is in a cycle,
based on the Hilbert Transform dominant cycle analysis.

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
    pd.Series: New feature generated (0 or 1).
"""
