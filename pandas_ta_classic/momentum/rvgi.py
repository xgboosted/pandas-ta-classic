# Relative Vigor Index (RVGI)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.swma import swma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def rvgi(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    swma_length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Relative Vigor Index (RVGI)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    swma_length = _pos_int(swma_length, 4, "swma_length")
    _length = max(length, swma_length)
    open_ = verify_series(open_, _length)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Both stay exact differences. An epsilon in the denominator survives the
    # swma and the rolling sum as dust rather than a zero, so a window with no
    # range at all divided by ~1e-15 and RVGI reached 1e14; an exact zero lets
    # the mask below mark it instead. In the numerator an epsilon only ever
    # reported movement a bar did not make.
    high_low_range = high - low
    close_open_range = close - open_

    # Calculate Result
    numerator = swma(close_open_range, length=swma_length).rolling(length).sum()
    denominator = swma(high_low_range, length=swma_length).rolling(length).sum()

    # A window with no range at all reads 0.0, the convention TA-Lib applies for
    # a degenerate window; see tests/test_degenerate_input.py. On consistent OHLC
    # the numerator vanishes with the divisor, because high == low forces
    # open == close; a malformed bar leaves it alive, and 0.0 is the marker there
    # too rather than eps/eps == 1.0, RVGI's most bullish reading.
    rvgi = (numerator / denominator.where(denominator != 0)).mask(denominator == 0, 0.0)
    signal = swma(rvgi, length=swma_length)
    histogram = rvgi - signal

    # Offset
    rvgi, signal, histogram = apply_offset([rvgi, signal, histogram], offset)

    # Handle fills
    rvgi, signal, histogram = apply_fill([rvgi, signal, histogram], **kwargs)

    # Name & Category
    rvgi.name = f"RVGI_{length}_{swma_length}"
    signal.name = f"RVGIs_{length}_{swma_length}"
    histogram.name = f"RVGIh_{length}_{swma_length}"
    rvgi.category = signal.category = histogram.category = "momentum"

    # Prepare DataFrame to return
    df = DataFrame({histogram.name: histogram, rvgi.name: rvgi, signal.name: signal})
    df.name = f"RVGI_{length}_{swma_length}"
    df.category = rvgi.category

    return df


rvgi.__doc__ = """Relative Vigor Index (RVGI)

The Relative Vigor Index attempts to measure the strength of a trend relative to
its closing price to its trading range.  It is based on the belief that it tends
to close higher than they open in uptrends or close lower than they open in
downtrends.

Sources:
    https://www.investopedia.com/terms/r/relative_vigor_index.asp

Calculation:
    Default Inputs:
        length=14, swma_length=4
    SWMA = Symmetrically Weighted Moving Average
    numerator = SUM(SWMA(close - open, swma_length), length)
    denominator = SUM(SWMA(high - low, swma_length), length)
    RVGI = numerator / denominator

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    swma_length (int): It's period. Default: 4
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
