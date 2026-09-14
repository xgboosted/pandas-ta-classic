# Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)
from typing import Any

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param


def ht_trendmode(
    close: Series,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Hilbert Transform - Trend vs Cycle Mode"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import HT_TRENDMODE

        result = HT_TRENDMODE(close).astype(int)
    else:
        ht = hilbert_result(close, ht_start=37, lookback=63)
        result = Series(ht["trend_mode"], index=close.index)
        # TA-Lib reports 0 through its lookback of 63 bars (and any leading
        # NaN run before it).
        result.iloc[: ht["first_valid"] + 63] = 0.0
        # Past the lookback, NaN marks bars whose trend mode is undefined
        # because a non-finite input poisoned the recursion. Report them
        # instead of silently turning them into 0 ("cycle mode"); fully
        # defined input keeps the historical int dtype.
        if result.notna().all():
            result = result.astype(int)

    # Offset
    result = apply_offset(result, offset)

    result = apply_fill(result, **kwargs)

    # Name and Categorize it
    result.name = "HT_TRENDMODE"
    result.category = "cycles"

    return result


ht_trendmode.__doc__ = """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)

Returns 1 when the market is in a trend and 0 when it is in a cycle,
based on the Hilbert Transform dominant cycle analysis.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated (0 or 1).
"""
