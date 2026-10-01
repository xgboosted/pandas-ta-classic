# Chaikins Volatility (CVI)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import apply_fill, apply_offset, degenerate_div, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def cvi(
    high: Series,
    low: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Chaikins Volatility (CVI)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    hl = high - low
    ema_hl = ema(hl, length=length, talib=False)
    if ema_hl is None:
        return None

    # Bars that trade at a single price smooth to a zero high-low range, so
    # this divides 0/0. degenerate_div masks it to 0.0 (TA-Lib's degenerate
    # marker); a nonzero numerator over a zero denominator (a flat stretch
    # followed by a gap) is a real x/0 and reads inf.
    denominator = ema_hl.shift(length)
    cvi_ = 100 * degenerate_div(ema_hl - denominator, denominator)

    # Offset
    cvi_ = apply_offset(cvi_, offset)

    cvi_ = apply_fill(cvi_, **kwargs)

    # Name and Categorize it
    cvi_.name = f"CVI_{length}"
    cvi_.category = "volatility"

    return cvi_


cvi.__doc__ = """Chaikins Volatility (CVI)

Chaikins Volatility measures the range between the high and low prices by
calculating the rate of change of the exponential moving average of the
High-Low spread. Rising CVI indicates expanding volatility; falling CVI
indicates contracting volatility.

HL = High - Low
EMA_HL = EMA(HL, length)
CVI = 100 * (EMA_HL - EMA_HL[length]) / EMA_HL[length]

Sources:
    Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:chaikins_volatility

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    length (int): EMA period and lookback. Default: 10
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: CVI values.
"""
