# Relative Strength Xtra (RSX)
from typing import Any, Optional, Union
import numpy as np
from pandas import concat, DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.utils import (
    apply_offset,
    get_drift,
    get_offset,
    signals,
    verify_series,
)


def rsx(
    close: Series,
    length: Optional[int] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Union[Series, DataFrame]]:
    """Indicator: Relative Strength Xtra (inspired by Jurik RSX)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    from pandas_ta_classic.utils._numba import _rsx_loop

    c_arr = close.to_numpy()
    m = close.size
    result = _rsx_loop(c_arr, length, m)
    rsx = Series(result, index=close.index)

    # Offset
    rsx = apply_offset(rsx, offset, **kwargs)

    # Name and Categorize it
    rsx.name = f"RSX_{length}"
    rsx.category = "momentum"

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = concat(
            [
                DataFrame({rsx.name: rsx}),
                signals(
                    indicator=rsx,
                    xa=kwargs.pop("xa", 80),
                    xb=kwargs.pop("xb", 20),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return rsx


rsx.__doc__ = """Relative Strength Xtra (rsx)

The Relative Strength Xtra is based on the popular RSI indicator and inspired
by the work Jurik Research. The code implemented is based on published code
found at 'prorealcode.com'. This enhanced version of the rsi reduces noise and
provides a clearer, only slightly delayed insight on momentum and velocity of
price movements.

Sources:
    http://www.jurikres.com/catalog1/ms_rsx.htm
    https://www.prorealcode.com/prorealtime-indicators/jurik-rsx/

Calculation:
    Refer to the sources above for information as well as code example.

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
