import numpy as np
import pandas as pd

from pandas_ta_classic.overlap import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int

# - Standard definition of your custom indicator function (including docs)-


def ni(close, length=None, centered=False, offset=None, **kwargs):
    """
    Example indicator ni
    """
    # Validate Arguments: None selects the default, anything else invalid raises
    length = _pos_int(length, 20, "length")
    centered = _bool_param(centered, False, "centered")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return

    # Calculate Result
    t = int(0.5 * length) + 1
    ma = sma(close, length)

    ni = close - ma.shift(t)
    if centered:
        ni = (close.shift(t) - ma).shift(-t)

    # Offset
    ni = apply_offset(ni, offset)

    # Handle fills: fillna=<value>, fill_method="ffill" or "bfill"
    ni = apply_fill(ni, **kwargs)

    # Name and Categorize it
    ni.name = f"ni_{length}"
    ni.category = "trend"

    return ni


ni.__doc__ = """Example indicator (NI)

Is an indicator provided solely as an example

Sources:
    https://github.com/xgboosted/pandas-ta-classic/issues/264

Calculation:
    Default Inputs:
        length=20, centered=False
    SMA = Simple Moving Average
    t = int(0.5 * length) + 1

    ni = close.shift(t) - SMA(close, length)
    if centered:
        ni = ni.shift(-t)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    centered (bool): Shift the ni back by int(0.5 * length) + 1. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): "ffill" or "bfill"

Returns:
    pd.Series: New feature generated.
"""

# - Define a matching class method --------------------------------------------


def ni_method(self, length=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = ni(close=close, length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)


# Demonstration of the custom indicator
if __name__ == "__main__":
    print("Testing custom NI (Example Indicator) function...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates, name="close")

    # Calculate the NI indicator
    result = ni(close_prices, length=20)

    print(f"Sample data shape: {close_prices.shape}")
    print(f"NI indicator shape: {result.shape}")
    print(f"NI indicator name: {result.name}")
    print("First 5 values:")
    print(result.head())
    print("Last 5 values:")
    print(result.tail())
    print("Custom NI indicator test completed successfully!")
