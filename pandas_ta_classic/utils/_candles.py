import numpy as np
from pandas import Series


def candle_color(open_: Series, close: Series) -> Series:
    """Classify each bar as bullish (+1) or bearish (-1).

    Args:
        open_: Series of 'open's.
        close: Series of 'close's.

    Returns:
        Series: ``1`` where ``close >= open_``, ``-1`` where ``close < open_``.
        A bar whose open or close is NaN has no defined colour and yields NaN,
        which makes the result float-typed; when every bar is defined the
        result keeps the historical integer dtype.
    """
    color = Series(np.nan, index=close.index)
    color[close >= open_] = 1.0
    color[close < open_] = -1.0
    if color.notna().all():
        return color.astype(int)
    return color
