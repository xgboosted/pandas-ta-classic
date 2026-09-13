# Candle Z (CDL_Z)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.statistics.zscore import zscore
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def _anchored_zscore(series: Series) -> Series:
    """Z Score of each bar against every bar up to and including it.

    Expanding rather than whole-sample: bar ``t`` is standardised with the mean
    and standard deviation of bars ``0..t`` only, so the value at ``t`` never
    depends on a later bar.  The first bar has no standard deviation and stays
    ``NaN``.
    """
    # pandas' expanding moments, not the pure-numpy style of statistics/zscore.py:
    # that one slides a fixed window, where numpy is exact and trivial.  Over an
    # expanding window the cumulative-sum equivalent loses precision to
    # cancellation (measured ~1e-11 relative against a two-pass reference, versus
    # ~8e-13 here), and pandas' online algorithm also matches its NaN handling.
    expanding = series.expanding(min_periods=2)
    return (series - expanding.mean()) / expanding.std(ddof=1)


def cdl_z(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    full: Optional[bool] = None,
    ddof: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Candle Type - Z Score"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    ddof = int(ddof) if ddof and ddof >= 0 and ddof < length else 1
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    full = bool(full) if full is not None and full else False

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    if full:
        # Anchored, not whole-sample: the previous implementation set the
        # window to close.size and then back-filled, which copied the final
        # bar's Z Score onto every earlier row -- a constant column built out
        # of future data.
        z_open = _anchored_zscore(open_)
        z_high = _anchored_zscore(high)
        z_low = _anchored_zscore(low)
        z_close = _anchored_zscore(close)
    else:
        z_open = zscore(open_, length=length, ddof=ddof)
        z_high = zscore(high, length=length, ddof=ddof)
        z_low = zscore(low, length=length, ddof=ddof)
        z_close = zscore(close, length=length, ddof=ddof)
        if z_open is None or z_high is None or z_low is None or z_close is None:
            return None

    _full = "a" if full else ""
    _props = _full if full else f"_{length}_{ddof}"
    df = DataFrame(
        {
            f"open_Z{_props}": z_open,
            f"high_Z{_props}": z_high,
            f"low_Z{_props}": z_low,
            f"close_Z{_props}": z_close,
        }
    )

    # Offset
    df = apply_offset(df, offset)

    df = apply_fill(df, **kwargs)

    # Name and Categorize it
    df.name = f"CDL_Z{_props}"
    df.category = "candles"

    return df


cdl_z.__doc__ = """Candle Type: Z

Normalizes OHLC Candles with a rolling Z Score.

Source: Kevin Johnson

Calculation:
    Default values:
        length=30, full=False, ddof=1
    Z = ZSCORE

    open  = Z( open, length, ddof)
    high  = Z( high, length, ddof)
    low   = Z(  low, length, ddof)
    close = Z(close, length, ddof)

    With full=True the rolling window is replaced by an anchored (expanding)
    one: each bar is standardised against every bar up to and including it.
    The first bar has no standard deviation and stays NaN.

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 10
    full (bool): Use an anchored (expanding) window instead of a rolling one
        of `length` bars. Default: False

Kwargs:
    naive (bool, optional): If True, prefills potential Doji less than
        the length if less than a percentage of it's high-low range.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: CDL_DOJI column.
"""
