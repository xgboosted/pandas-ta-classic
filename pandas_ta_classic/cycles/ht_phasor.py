# -*- coding: utf-8 -*-
# Hilbert Transform - Phasor Components (HT_PHASOR)
from typing import Any, Optional

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_offset,
    verify_series,
)


def ht_phasor(
    close: Series,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Hilbert Transform - Phasor Components"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import HT_PHASOR as taHT

        inphase_arr, quad_arr = taHT(close)
        inphase = Series(inphase_arr, index=close.index)
        quadrature = Series(quad_arr, index=close.index)
    else:
        ht = hilbert_result(close)
        inphase = Series(ht["in_phase"], index=close.index)
        quadrature = Series(ht["quadrature"], index=close.index)

    return _build_dataframe(
        {"HT_PHASOR_INPHASE": inphase, "HT_PHASOR_QUAD": quadrature},
        "HT_PHASOR",
        "cycles",
        offset,
        **kwargs,
    )


ht_phasor.__doc__ = """Hilbert Transform - Phasor Components (HT_PHASOR)

Returns the InPhase and Quadrature components of the Hilbert Transform,
which together form a phasor representation of the dominant cycle.

Sources:
    John F. Ehlers, "Rocket Science for Traders"

Args:
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: inphase and quadrature columns.
"""
