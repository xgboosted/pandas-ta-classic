# -*- coding: utf-8 -*-
# Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)
from typing import Any, Optional
import numpy as np
from pandas import Series
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import get_offset, verify_series


def ht_trendmode(
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Trend vs Cycle Mode"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ht = hilbert_result(close, ht_start=37)
    result = Series(ht["trend_mode"], index=close.index)
    # TA-Lib lookback for HT_TRENDMODE is 63; the Hilbert variables
    # have not converged before that.  Blank the warmup zone so the
    # fillna below converts them to 0, matching TA-Lib output.
    result.iloc[:63] = np.nan

    # Convert to int, treating NaN as 0 to match TA-Lib output
    result = result.fillna(-1).astype(int).replace(-1, 0)

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            result.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            result.bfill(inplace=True)

    # Name and Categorize it
    result.name = "HT_TRENDMODE"
    result.category = "cycles"

    return result


ht_trendmode.__doc__ = """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)

Returns 1 when the market is in a trend and 0 when it is in a cycle,
based on the Hilbert Transform dominant cycle analysis.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated (0 or 1).
"""
