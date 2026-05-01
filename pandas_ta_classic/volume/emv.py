# -*- coding: utf-8 -*-
# Ease of Movement (EMV)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    non_zero_range,
    verify_series,
)


def emv(
    high: Series,
    low: Series,
    volume: Series,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Ease of Movement (EMV)

    Raw (unsmoothed) Ease of Movement oscillator.
    Formula: ((H+L)/2 - prev(H+L)/2) / (Volume / (H-L))

    tulipy name: EMV.  See also ta.eom which applies SMA smoothing.
    """
    high = verify_series(high)
    low = verify_series(low)
    volume = verify_series(volume)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or volume is None:
        return None

    # divisor=10000 matches tulipy's EMV scaling convention
    divisor = kwargs.pop("divisor", 10000)
    hl_range = non_zero_range(high, low)
    midpoint = 0.5 * (high + low)
    distance = midpoint - midpoint.shift(drift)
    box_ratio = (volume / divisor) / hl_range
    result = distance / box_ratio

    # Offset
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)

    result.name = "EMV"
    result.category = "volume"
    return result


emv.__doc__ = """Ease of Movement (EMV)

Raw (unsmoothed) Ease of Movement.  Higher values indicate price is moving
up with ease on low volume; lower values indicate the opposite.

Formula:
    midpoint  = (High + Low) / 2
    distance  = midpoint - prev(midpoint)
    box_ratio = Volume / (High - Low)
    EMV       = distance / box_ratio

See ta.eom for the SMA-smoothed version.
tulipy name: EMV.

Args:
    high (pd.Series): Series of 'high' prices
    low (pd.Series): Series of 'low' prices
    volume (pd.Series): Series of volume
    drift (int): Difference period. Default: 1
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
