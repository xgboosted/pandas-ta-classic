# -*- coding: utf-8 -*-
# Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def ht_dcphase(
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Hilbert Transform - Dominant Cycle Phase"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ht = hilbert_result(close, ht_start=37)
    result = Series(ht["dc_phase"], index=close.index)

    # Offset
    result = apply_offset(result, offset)

    result = apply_fill(result, **kwargs)

    # Name and Categorize it
    result.name = "HT_DCPHASE"
    result.category = "cycles"

    return result


ht_dcphase.__doc__ = """Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)

The Dominant Cycle Phase uses the Hilbert Transform to estimate the
phase of the dominant cycle.

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
