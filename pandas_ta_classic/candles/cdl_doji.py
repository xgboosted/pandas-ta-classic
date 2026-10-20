# Candle Doji (CDL_DOJI)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.candles._cdl_math import period_average
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_int, nan_on_short_input


@nan_on_short_input
def cdl_doji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    factor: float | None = None,
    scalar: float | None = None,
    asint: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Candle Type - Doji"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    asint = _bool_param(asint, True, "asint")
    factor = _number(factor, 10, "factor", ge=0)
    scalar = _number(scalar, 100, "scalar")
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    naive = _bool_param(kwargs.pop("naive", None), False, "naive")

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib's CDLDOJI: the body is compared with ``factor`` percent of the mean
    # high-low range of the *previous* ``length`` bars, using TA-Lib's
    # arithmetic (TA_CANDLEAVERAGE) so a body exactly on the threshold is judged
    # as TA-Lib judges it. A flat bar has range and body 0, not epsilon, and is
    # a doji (0 <= 0), as in TA-Lib.
    body = (close - open_).abs()
    hl_range = high - low
    # Average the previous ``length`` finite bars, as if NaN rows were dropped:
    # a row resample() inserts for a missing session would otherwise leave the
    # next ``length`` averages NaN, and weekend gaps keep every average NaN.
    # An inf row is dropped too, as in run_pattern(): inside the running total
    # it would make every later average inf and then NaN (inf - inf).
    finite = np.isfinite(np.vstack([s.to_numpy(dtype=float) for s in (open_, high, low, close)])).all(axis=0)
    threshold = np.full(len(close), np.nan)
    threshold[finite] = period_average(hl_range.to_numpy(dtype=float)[finite], length, factor / 100, 1.0, 0, length)
    doji = body <= Series(threshold, index=close.index)

    if naive:
        # Only bars without ``length`` previous finite bars -- the first
        # ``length`` finite ones -- fall back to their own high-low range.
        # Counted, not read off a NaN threshold: an inf range turns every
        # later running total into NaN, and those bars do have a window.
        no_window = finite & (np.cumsum(finite) <= length)
        doji.iloc[no_window] = (body <= factor / 100 * hl_range).to_numpy()[no_window]
    if asint:
        # float64, as every native pattern returns; an int scalar used to make this int64,
        # so cdl_pattern's frame mixed one int64 column into float64 ones
        doji = doji.astype(float) * scalar

    # Offset
    doji = apply_offset(doji, offset)

    doji = apply_fill(doji, **kwargs)

    # Name and Categorize it
    doji.name = f"CDL_DOJI_{length}_{0.01 * factor}"
    doji.category = "candles"

    return doji


cdl_doji.__doc__ = """Candle Type: Doji

A candle body is Doji, when it is no longer than 10% of the
average of the 10 previous candles' high-low range.

Sources:
    TA-Lib: CDLDOJI, bar for bar on finite input. Other ``length`` and
    ``factor`` values match TA-Lib with its BodyDoji setting changed to
    (HighLow, length, factor / 100). A row with a NaN or infinite price
    reports 0 and is left out of the average, as in the other candle
    patterns; TA-Lib keeps an infinite open or close row in its average.

Calculation:
    Default values:
        length=10, factor=10 (0.1), scalar=100
    ABS = Absolute Value
    TOTAL = running sum of HL_RANGE over the previous length bars,
        seeded left to right and updated as TA-Lib does

    BODY = ABS(close - open)
    HL_RANGE = high - low

    DOJI = scalar IF BODY <= (factor / 100) * (TOTAL / length) ELSE 0

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 10
    factor (float): Doji value, as a percent of the high-low range. Default: 10
    scalar (float): How much to magnify. Default: 100
    asint (bool): Keep results numerical instead of boolean. Default: True

Kwargs:
    naive (bool, optional): If True, the first ``length`` finite bars,
        which have no window of previous bars, are compared with
        ``factor`` percent of their own high-low range instead of
        reporting 0. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: CDL_DOJI column.
"""
