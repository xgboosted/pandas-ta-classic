# -*- coding: utf-8 -*-
# Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def ht_trendline(
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Instantaneous Trendline"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ht = hilbert_result(close, ht_start=37)
    result = Series(ht["trendline"], index=close.index)

    # Offset
    result = apply_offset(result, offset)

    result = apply_fill(result, **kwargs)

    # Name and Categorize it
    result.name = "HT_TRENDLINE"
    result.category = "overlap"

    return result


ht_trendline.__doc__ = """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE)

The Instantaneous Trendline uses the Hilbert Transform dominant cycle
period to compute a smoothed trendline that adapts to the current
market cycle.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
