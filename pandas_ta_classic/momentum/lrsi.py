# Laguerre Relative Strength Index (Laguerre RSI)
from typing import Any, Optional
from numpy import maximum, where, zeros
from pandas import Series
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def lrsi(
    close: Series,
    length: Optional[int] = None,
    gamma: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Laguerre RSI (LRSI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    gamma = float(gamma) if gamma and 0 < gamma < 1 else 0.5
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    from pandas_ta_classic.utils._numba import _lrsi_loop

    close_arr = close.values
    n = len(close)

    l0, l1, l2, l3 = _lrsi_loop(close_arr, n, gamma)

    # Calculate Laguerre RSI components (can be vectorized)
    cu = zeros(n)
    cd = zeros(n)

    # Vectorized calculation of up/down moves between filter stages
    cu += maximum(l0 - l1, 0)
    cd += maximum(l1 - l0, 0)
    cu += maximum(l1 - l2, 0)
    cd += maximum(l2 - l1, 0)
    cu += maximum(l2 - l3, 0)
    cd += maximum(l3 - l2, 0)

    # Calculate LRSI with division by zero protection
    denominator = cu + cd
    # Replace zeros with 1 to avoid division by zero (result will be 0 anyway since cu=0 when denominator=0)
    denominator = where(denominator == 0, 1, denominator)
    lrsi = Series(100 * cu / denominator, index=close.index)

    return _finalize(lrsi, offset, f"LRSI_{length}", "momentum", **kwargs)


lrsi.__doc__ = """Laguerre RSI (LRSI)

The Laguerre RSI is a modified RSI indicator that uses Laguerre polynomials to
reduce lag and provide earlier signals. It adapts to price changes more quickly
than the standard RSI while maintaining smooth oscillations.

Sources:
    https://www.tradingview.com/script/3p0QrN5C-Laguerre-RSI/
    https://www.mesasoftware.com/papers/LaguerreFilters.pdf

Calculation:
    Default Inputs:
        length=14, gamma=0.5

    Apply Laguerre filter with gamma coefficient:
    L0 = (1 - gamma) * Close + gamma * L0[1]
    L1 = -gamma * L0 + L0[1] + gamma * L1[1]
    L2 = -gamma * L1 + L1[1] + gamma * L2[1]
    L3 = -gamma * L2 + L2[1] + gamma * L3[1]

    Calculate ups and downs:
    CU = sum of (L0-L1, L1-L2, L2-L3) when positive
    CD = sum of (L0-L1, L1-L2, L2-L3) when negative (absolute)

    LRSI = 100 * CU / (CU + CD)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    gamma (float): Laguerre filter coefficient (0 to 1). Default: 0.5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
