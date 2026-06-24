# Relative Volatility Index (RVI)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.statistics import stdev
from pandas_ta_classic.utils import apply_fill, apply_offset, get_drift, get_offset
from pandas_ta_classic.utils import unsigned_differences, verify_series


def _pos_int(val, default):
    return int(val) if val and val > 0 else default


def _pos_float(val, default):
    return float(val) if val and val > 0 else default


def _rvi_compute(source: Series, length: int, scalar: float, mode: str, drift: int) -> Optional[Series]:
    """Core RVI computation for a single source series."""
    std = stdev(source, length)
    pos, neg = unsigned_differences(source, amount=drift)
    pos_std = pos * std
    neg_std = neg * std
    pos_avg = ma(mode, pos_std, length=length)
    if pos_avg is None:
        return None
    neg_avg = ma(mode, neg_std, length=length)
    if neg_avg is None:
        return None
    result = scalar * pos_avg
    result /= pos_avg + neg_avg
    return result


def _rvi_mode(refined, thirds, high, low, close, length, scalar, mamode, drift):
    """Compute RVI for the requested mode; returns (series, mode_str) or None."""
    if refined:
        high_rvi = _rvi_compute(high, length, scalar, mamode, drift)
        if high_rvi is None:
            return None
        low_rvi = _rvi_compute(low, length, scalar, mamode, drift)
        if low_rvi is None:
            return None
        return 0.5 * (high_rvi + low_rvi), "r"
    if thirds:
        high_rvi = _rvi_compute(high, length, scalar, mamode, drift)
        if high_rvi is None:
            return None
        low_rvi = _rvi_compute(low, length, scalar, mamode, drift)
        if low_rvi is None:
            return None
        close_rvi = _rvi_compute(close, length, scalar, mamode, drift)
        if close_rvi is None:
            return None
        return (high_rvi + low_rvi + close_rvi) / 3.0, "t"
    result = _rvi_compute(close, length, scalar, mamode, drift)
    return (result, "") if result is not None else None


def rvi(
    close: Series,
    high: Optional[Series] = None,
    low: Optional[Series] = None,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    refined: Optional[bool] = None,
    thirds: Optional[bool] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Relative Volatility Index (RVI)"""
    # Validate arguments
    length = _pos_int(length, 14)
    scalar = _pos_float(scalar, 100)
    refined = bool(refined)
    thirds = bool(thirds)
    mamode = mamode if isinstance(mamode, str) else "ema"
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    if refined or thirds:
        high = verify_series(high)
        low = verify_series(low)
        if high is None or low is None:
            return None

    # Calculate Result
    result = _rvi_mode(refined, thirds, high, low, close, length, scalar, mamode, drift)
    if result is None:
        return None
    rvi, _mode = result

    # Offset
    rvi = apply_offset(rvi, offset)

    rvi = apply_fill(rvi, **kwargs)

    # Name and Categorize it
    rvi.name = f"RVI{_mode}_{length}"
    rvi.category = "volatility"

    return rvi


rvi.__doc__ = """Relative Volatility Index (RVI)

The Relative Volatility Index (RVI) was created in 1993 and revised in 1995.
Instead of adding up price changes like RSI based on price direction, the RVI
adds up standard deviations based on price direction.

Sources:
    https://www.tradingview.com/wiki/Keltner_Channels_(KC)

Calculation:
    Default Inputs:
        length=14, scalar=100, refined=None, thirds=None
    EMA = Exponential Moving Average
    STDEV = Standard Deviation

    UP = STDEV(src, length) IF src.diff() > 0 ELSE 0
    DOWN = STDEV(src, length) IF src.diff() <= 0 ELSE 0

    UPSUM = EMA(UP, length)
    DOWNSUM = EMA(DOWN, length)

    RVI = scalar * (UPSUM / (UPSUM + DOWNSUM))

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 14
    scalar (float): A positive float to scale the bands. Default: 100
    refined (bool): Use 'refined' calculation which is the average of
        RVI(high) and RVI(low) instead of RVI(close). Default: False
    thirds (bool): Average of high, low and close. Default: False
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, basis, upper columns.
"""
