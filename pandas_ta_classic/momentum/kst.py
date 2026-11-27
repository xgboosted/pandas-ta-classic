# Know Sure Thing (KST)
from typing import Annotated, Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import Bound, _name_tokens, _positive_inputs, nan_on_short_input

from .roc import roc


@nan_on_short_input
@_positive_inputs("close")
def kst(
    close: Series,
    roc1: Annotated[int, Bound(gt=0)] = 10,
    roc2: Annotated[int, Bound(gt=0)] = 15,
    roc3: Annotated[int, Bound(gt=0)] = 20,
    roc4: Annotated[int, Bound(gt=0)] = 30,
    sma1: Annotated[int, Bound(gt=0)] = 10,
    sma2: Annotated[int, Bound(gt=0)] = 10,
    sma3: Annotated[int, Bound(gt=0)] = 10,
    sma4: Annotated[int, Bound(gt=0)] = 15,
    signal: Annotated[int, Bound(gt=0)] = 9,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: 'Know Sure Thing' (KST)"""
    # Validate arguments

    _length = max(roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    rocma1 = roc(close, roc1).rolling(sma1).mean()
    rocma2 = roc(close, roc2).rolling(sma2).mean()
    rocma3 = roc(close, roc3).rolling(sma3).mean()
    rocma4 = roc(close, roc4).rolling(sma4).mean()

    kst = rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4
    kst_signal = kst.rolling(signal).mean()

    # Offset
    kst, kst_signal = apply_offset([kst, kst_signal], offset)

    kst, kst_signal = apply_fill([kst, kst_signal], **kwargs)

    # Name and Categorize it
    kst.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}"
    # The signal line smooths the KST line, so its periods name it too.
    kst_signal.name = f"KSTs_{signal}" + _name_tokens(
        ("roc1", roc1, 10),
        ("roc2", roc2, 15),
        ("roc3", roc3, 20),
        ("roc4", roc4, 30),
        ("sma1", sma1, 10),
        ("sma2", sma2, 10),
        ("sma3", sma3, 10),
        ("sma4", sma4, 15),
    )
    kst.category = kst_signal.category = "momentum"

    # Prepare DataFrame to return
    data = {kst.name: kst, kst_signal.name: kst_signal}
    kstdf = DataFrame(data)
    kstdf.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}_{signal}"
    kstdf.category = "momentum"

    return kstdf


kst.__doc__ = """'Know Sure Thing' (KST)

The 'Know Sure Thing' is a momentum based oscillator and based on ROC.

Sources:
    https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
    https://www.incrediblecharts.com/indicators/kst.php

Calculation:
    Default Inputs:
        roc1=10, roc2=15, roc3=20, roc4=30,
        sma1=10, sma2=10, sma3=10, sma4=15, signal=9
    ROC = Rate of Change
    SMA = Simple Moving Average
    rocsma1 = SMA(ROC(close, roc1), sma1)
    rocsma2 = SMA(ROC(close, roc2), sma2)
    rocsma3 = SMA(ROC(close, roc3), sma3)
    rocsma4 = SMA(ROC(close, roc4), sma4)

    KST = rocsma1 + 2 * rocsma2 + 3 * rocsma3 + 4 * rocsma4
    KST_Signal = SMA(KST, signal)

Args:
    close (pd.Series): Series of 'close's
    roc1 (int): ROC 1 period. Default: 10
    roc2 (int): ROC 2 period. Default: 15
    roc3 (int): ROC 3 period. Default: 20
    roc4 (int): ROC 4 period. Default: 30
    sma1 (int): SMA 1 period. Default: 10
    sma2 (int): SMA 2 period. Default: 10
    sma3 (int): SMA 3 period. Default: 10
    sma4 (int): SMA 4 period. Default: 15
    signal (int): It's period. Default: 9
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: kst and kst_signal columns
"""
