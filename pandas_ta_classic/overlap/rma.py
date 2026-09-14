# Wilder's Moving Average (RMA)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def rma(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: wildeR's Moving Average (RMA)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    alpha = (1.0 / length) if length > 0 else 0.5
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result — SMA-seeded Wilder smoothing (matches TA-Lib)
    close = close.copy()
    # Seed on the first `length` values after any leading NaN run, as ema()
    # does. Seeding on iloc[0:length] averaged whatever valid values the NaN
    # prefix left (chained input: dx, the DM series), so every value derived
    # from that seed started from the wrong level.
    first_valid = close.first_valid_index()
    fv_pos = None if first_valid is None else close.index.get_loc(first_valid)
    if fv_pos is None or fv_pos + length > close.size:
        # No SMA seed exists, so the average is undefined at every position.
        rma = Series(np.nan, index=close.index)
    else:
        sma_nth = close.iloc[fv_pos : fv_pos + length].mean()
        close.iloc[: fv_pos + length - 1] = np.nan
        close.iloc[fv_pos + length - 1] = sma_nth
        rma = close.ewm(alpha=alpha, adjust=False).mean()

    # Offset
    rma = apply_offset(rma, offset)

    rma = apply_fill(rma, **kwargs)

    # Name & Category
    rma.name = f"RMA_{length}"
    rma.category = "overlap"

    return rma


rma.__doc__ = """Wilder's Moving Average (RMA)

Wilder's Moving Average is simply an Exponential Moving Average (EMA) with
a modified alpha = 1 / length.

Sources:
    https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
    https://www.incrediblecharts.com/indicators/wilder_moving_average.php

Calculation:
    Default Inputs:
        length=10
    alpha = 1 / length
    SMA_nth = SMA(close, length)
    close[:length - 1] = NaN
    close[length - 1] = SMA_nth
    RMA = EWM(close, alpha=alpha, adjust=False)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
