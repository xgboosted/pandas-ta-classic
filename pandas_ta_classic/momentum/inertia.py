# Inertia (INERTIA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.volatility import rvi
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def _pos_int(val, default):
    """Return ``int(val)`` when *val* is a positive integer, else *default*."""
    return int(val) if val and val > 0 else default


def _pos_float(val, default):
    """Return ``float(val)`` when *val* is a positive float, else *default*."""
    return float(val) if val and val > 0 else default


def _inertia_rvi_mode(close, high, low, rvi_length, scalar, refined, thirds, mamode):
    """Compute RVI and return a ``(mode_suffix, rvi_series)`` tuple.

    Selects the correct :func:`rvi` call based on *refined* / *thirds* flags
    and returns a short string suffix (``"r"``, ``"t"``, or ``""``) together
    with the resulting series.

    Args:
        close (Series): Close price series.
        high (Series | None): High price series (required for refined/thirds).
        low (Series | None): Low price series (required for refined/thirds).
        rvi_length (int): Look-back period for RVI.
        scalar (float): Scalar multiplier for RVI.
        refined (bool): Use refined RVI variant.
        thirds (bool): Use thirds RVI variant.
        mamode (str): Moving-average mode passed to :func:`rvi`.

    Returns:
        tuple[str, Series | None]: ``(mode_suffix, rvi_series)``.
    """
    if refined:
        return "r", rvi(
            close,
            high=high,
            low=low,
            length=rvi_length,
            scalar=scalar,
            refined=refined,
            mamode=mamode,
        )
    if thirds:
        return "t", rvi(
            close,
            high=high,
            low=low,
            length=rvi_length,
            scalar=scalar,
            thirds=thirds,
            mamode=mamode,
        )
    return "", rvi(close, length=rvi_length, scalar=scalar, mamode=mamode)


def inertia(
    close: Optional[Series] = None,
    high: Optional[Series] = None,
    low: Optional[Series] = None,
    length: Optional[int] = None,
    rvi_length: Optional[int] = None,
    scalar: Optional[float] = None,
    refined: Optional[bool] = None,
    thirds: Optional[bool] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Inertia (INERTIA)"""
    # Validate Arguments
    length = _pos_int(length, 20)
    rvi_length = _pos_int(rvi_length, 14)
    scalar = _pos_float(scalar, 100)
    refined = bool(refined)
    thirds = bool(thirds)
    mamode = mamode if isinstance(mamode, str) else "ema"
    _length = max(length, rvi_length)
    close = verify_series(close, _length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    if refined or thirds:
        high = verify_series(high, _length)
        low = verify_series(low, _length)
        if high is None or low is None:
            return None

    # Calculate Result
    _mode, rvi_ = _inertia_rvi_mode(close, high, low, rvi_length, scalar, refined, thirds, mamode)

    if rvi_ is None:
        return None
    inertia = linreg(rvi_, length=length)
    if inertia is None:
        return None

    # Offset
    inertia = apply_offset(inertia, offset)

    inertia = apply_fill(inertia, **kwargs)

    # Name & Category
    _props = f"_{length}_{rvi_length}"
    inertia.name = f"INERTIA{_mode}{_props}"
    inertia.category = "momentum"

    return inertia


inertia.__doc__ = """Inertia (INERTIA)

Inertia was developed by Donald Dorsey and was introduced his article
in September, 1995. It is the Relative Vigor Index smoothed by the Least
Squares Moving Average. Postive Inertia when values are greater than 50,
Negative Inertia otherwise.

Sources:
    https://www.investopedia.com/terms/r/relative_vigor_index.asp

Calculation:
    Default Inputs:
        length=14, ma_length=20
    LSQRMA = Least Squares Moving Average

    INERTIA = LSQRMA(RVI(length), ma_length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    rvi_length (int): RVI period. Default: 14
    refined (bool): Use 'refined' calculation. Default: False
    thirds (bool): Use 'thirds' calculation. Default: False
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
