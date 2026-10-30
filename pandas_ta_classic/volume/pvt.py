# Price Volume Trend (PVT)
from typing import Any

from pandas import Series

from pandas_ta_classic.momentum.roc import roc
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def pvt(
    close: Series,
    volume: Series,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Price-Volume Trend (PVT)"""
    # Validate arguments
    close = verify_series(close)
    volume = verify_series(volume)
    if close is None or volume is None:
        return None
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    # The fractional change, as in TA-Lib's PVT; a percent made it 100 times larger (#344).
    roc_ = roc(close=close, length=drift, scalar=1)
    if roc_ is None:
        # Fewer rows than drift: there is no rate of change to accumulate.
        return None
    pv = roc_ * volume
    pvt = pv.cumsum()

    # Offset
    pvt = apply_offset(pvt, offset)

    pvt = apply_fill(pvt, **kwargs)

    # Name and Categorize it
    pvt.name = "PVT"
    pvt.category = "volume"

    return pvt


pvt.__doc__ = """Price-Volume Trend (PVT)

The Price-Volume Trend utilizes the Rate of Change with volume to
and it's cumulative values to determine money flow.

Sources:
    https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)

Calculation:
    Default Inputs:
        drift=1
    ROC = Rate of Change, as a fraction (scalar=1), as in TA-Lib's PVT
    pv = ROC(close, drift, scalar=1) * volume
    PVT = pv.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    drift (int): The diff period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
