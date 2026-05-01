# -*- coding: utf-8 -*-
# Hilbert Transform - SineWave (HT_SINE)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def ht_sine(
    close: Series,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Hilbert Transform - SineWave"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ht = hilbert_result(close, ht_start=37)
    sine = Series(ht["sine"], index=close.index)
    lead_sine = Series(ht["lead_sine"], index=close.index)

    # Offset
    sine, lead_sine = apply_offset([sine, lead_sine], offset)

    # Handle fills
    sine, lead_sine = apply_fill([sine, lead_sine], **kwargs)

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
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: sine and leadsine columns.
"""
