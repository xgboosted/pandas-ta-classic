# BRAR (Bull and Bear Ratio)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    degenerate_div,
    get_drift,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _number, _pos_int, nan_on_short_input


@nan_on_short_input
def brar(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: BRAR (BRAR)"""
    # Validate Arguments
    length = _pos_int(length, 26, "length")
    scalar = _number(scalar, 100, "scalar")
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # All four differences are exact: an epsilon in a numerator would make a
    # fully flat window divide eps / 0 and read inf instead of the 0.0 marker.
    # degenerate_div masks only a window where numerator and denominator are
    # both zero; a zero denominator with a positive numerator (a gap-up bar,
    # open == low, high > open) is a real x/0 and reads inf.
    high_open_range = high - open_
    open_low_range = open_ - low

    # Calculate Result
    hcy = high - close.shift(drift)
    cyl = close.shift(drift) - low

    hcy[hcy < 0] = 0  # Zero negative values
    cyl[cyl < 0] = 0  # ""

    olr_sum = open_low_range.rolling(length).sum()
    ar = scalar * degenerate_div(high_open_range.rolling(length).sum(), olr_sum)

    cyl_sum = cyl.rolling(length).sum()
    br = scalar * degenerate_div(hcy.rolling(length).sum(), cyl_sum)

    # Offset
    ar, br = apply_offset([ar, br], offset)

    ar, br = apply_fill([ar, br], **kwargs)

    # Name and Categorize it
    _props = f"_{length}"
    ar.name = f"AR{_props}"
    br.name = f"BR{_props}"
    ar.category = br.category = "momentum"

    # Prepare DataFrame to return
    brardf = DataFrame({ar.name: ar, br.name: br})
    brardf.name = f"BRAR{_props}"
    brardf.category = "momentum"

    return brardf


brar.__doc__ = """BRAR (BRAR)

BR and AR

Sources:
    No internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=26, scalar=100
    SUM = Sum

    HO_Diff = high - open
    OL_Diff = open - low
    HCY = high - close[-1]
    CYL = close[-1] - low
    HCY[HCY < 0] = 0
    CYL[CYL < 0] = 0
    AR = scalar * SUM(HO, length) / SUM(OL, length)
    BR = scalar * SUM(HCY, length) / SUM(CYL, length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 26
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ar, br columns.
"""
