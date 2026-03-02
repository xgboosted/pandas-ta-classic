# Candle Doji (CDL_DOJI)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import get_offset, high_low_range, is_percent
from pandas_ta_classic.utils import _finalize, real_body, verify_series


def cdl_doji(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    factor: Optional[float] = None,
    scalar: Optional[float] = None,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Candle Type - Doji"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    factor = float(factor) if is_percent(factor) else 10
    scalar = float(scalar) if scalar else 100
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    naive = kwargs.pop("naive", False)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    # TA-Lib averages the HL range of the *previous* ``length`` bars
    # (excluding the current bar), so shift the SMA by 1.
    body = real_body(open_, close).abs()
    hl_range = high_low_range(high, low).abs()
    hl_range_avg = sma(hl_range, length).shift(1)
    doji = body <= 0.01 * factor * hl_range_avg

    if naive:
        doji.iloc[:length] = body <= 0.01 * factor * hl_range
    if asint:
        doji = scalar * doji.astype(int)

    return _finalize(
        doji, offset, f"CDL_DOJI_{length}_{0.01 * factor}", "candles", **kwargs
    )


cdl_doji.__doc__ = """Candle Type: Doji

A candle body is Doji, when it's shorter than 10% of the
average of the 10 previous candles' high-low range.

Sources:
    TA-Lib CDL_DOJI C implementation

Calculation:
    Default values:
        length=10, factor=10 (percent=0.1), scalar=100
    ABS = Absolute Value
    SMA = Simple Moving Average

    BODY = ABS(close - open)
    HL_RANGE = ABS(high - low)
    AVG_RANGE = SMA(HL_RANGE, length).shift(1)   # previous bars only

    DOJI = scalar IF BODY <= factor/100 * AVG_RANGE ELSE 0

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 10
    factor (float): Doji value. Default: 100
    scalar (float): How much to magnify. Default: 100
    asint (bool): Keep results numerical instead of boolean. Default: True

Kwargs:
    naive (bool, optional): If True, prefills potential Doji less than
        the length if less than a percentage of it's high-low range.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: CDL_DOJI column.
"""
