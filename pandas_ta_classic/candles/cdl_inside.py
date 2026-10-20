# Candle Inside (CDL_INSIDE)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, candle_color, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _number, nan_on_short_input


@nan_on_short_input
def cdl_inside(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    asbool: bool | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Candle Type - Inside Bar"""
    # Validate arguments
    asbool = _bool_param(asbool, False, "asbool")
    scalar = _number(scalar, 100, "scalar")
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    # Compare with the previous finite bar, as if NaN rows were dropped, so the
    # bar after a resample() gap is still classified.
    finite = open_.notna() & high.notna() & low.notna() & close.notna()
    inside = Series(False, index=close.index)
    inside[finite.to_numpy()] = ((high[finite].diff() < 0) & (low[finite].diff() > 0)).to_numpy()

    if not asbool:
        # +-scalar (100), float64, as every pattern reads; it read +-1 and ignored scalar.
        # + 0.0 keeps a negative scalar from writing -0.0 on the other bars.
        inside = (inside * candle_color(open_, close)).astype(float) * scalar + 0.0

    # Offset
    inside = apply_offset(inside, offset)

    inside = apply_fill(inside, **kwargs)

    # Name and Categorize it
    inside.name = "CDL_INSIDE"
    inside.category = "candles"

    return inside


cdl_inside.__doc__ = """Candle Type: Inside Bar

An Inside Bar is a bar that is engulfed by the prior highs and lows of it's
previous bar. In other words, the current bar is smaller than it's previous bar.
Set asbool=True if you want to know if it is an Inside Bar. Note by default
asbool=False so this returns 0 if it is not an Inside Bar, scalar (100) if it is
an Inside Bar and close > open, and -scalar (-100) if it is an Inside Bar but
close < open, as the other candle patterns read.

Sources:
    https://www.tradingview.com/script/IyIGN1WO-Inside-Bar/

Calculation:
    Default Inputs:
        asbool=False
    inside = (high.diff() < 0) & (low.diff() > 0)

    if not asbool:
        inside *= scalar * candle_color(open_, close)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    asbool (bool): Returns the boolean result. Default: False
    scalar (float): How much to magnify. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""
