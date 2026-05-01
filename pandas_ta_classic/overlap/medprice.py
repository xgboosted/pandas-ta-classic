# -*- coding: utf-8 -*-
# Median Price (MEDPRICE)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def medprice(
    high: Series,
    low: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Median Price (MEDPRICE)

    Per-bar (High + Low) / 2.  Equivalent to ta.hl2 (per-bar).
    TA-Lib name: MEDPRICE.

    Note: distinct from ta.midprice which is a rolling indicator.
    """
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    result = 0.5 * (high + low)

    # Offset
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)

    result.name = "MEDPRICE"
    result.category = "overlap"
    return result


medprice.__doc__ = """Median Price (MEDPRICE)

MEDPRICE = (High + Low) / 2

Equivalent to ta.hl2.  TA-Lib name: MEDPRICE.

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
