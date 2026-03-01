# -*- coding: utf-8 -*-
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def td_seq(
    close: Series,
    asint: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Tom Demark Sequential (TD_SEQ)"""
    # Validate arguments
    close = verify_series(close)
    if close is None:
        return None
    offset = get_offset(offset)
    asint = asint if isinstance(asint, bool) else False
    show_all = kwargs.setdefault("show_all", True)

    def calc_td(series: Series, direction: str, show_all: bool):
        bool_arr = (
            (series.diff(4) > 0) if direction == "up" else (series.diff(4) < 0)
        ).to_numpy()

        # Vectorised consecutive-True streak, capped at 13 (equivalent to
        # rolling(13, min_periods=0).apply(true_sequence_count))
        n = len(bool_arr)
        streak: np.ndarray = np.zeros(n, dtype=float)
        for i in range(1, n):
            if bool_arr[i]:
                streak[i] = streak[i - 1] + 1 if streak[i - 1] < 13 else 13

        td_num = Series(streak, index=series.index)

        if show_all:
            td_num = td_num.mask(td_num == 0)
        else:
            td_num = td_num.mask(~td_num.between(6, 9))

        return td_num

    up_seq = calc_td(close, "up", show_all)
    down_seq = calc_td(close, "down", show_all)

    if asint:
        if up_seq.hasnans and down_seq.hasnans:
            up_seq.fillna(0, inplace=True)
            down_seq.fillna(0, inplace=True)
        up_seq = up_seq.astype(int)
        down_seq = down_seq.astype(int)

    # Offset
    up_seq = apply_offset(up_seq, offset, **kwargs)
    down_seq = apply_offset(down_seq, offset, **kwargs)

    # Name & Category
    up_seq.name = "TD_SEQ_UPa" if show_all else "TD_SEQ_UP"
    down_seq.name = "TD_SEQ_DNa" if show_all else "TD_SEQ_DN"
    up_seq.category = down_seq.category = "momentum"

    # Prepare Dataframe to return
    df = DataFrame({up_seq.name: up_seq, down_seq.name: down_seq})
    df.name = "TD_SEQ"
    df.category = up_seq.category

    return df


td_seq.__doc__ = """TD Sequential (TD_SEQ)

Tom DeMark's Sequential indicator attempts to identify a price point where an
uptrend or a downtrend exhausts itself and reverses.

Sources:
    https://tradetrekker.wordpress.com/tdsequential/

Calculation:
    Compare current close price with 4 days ago price, up to 13 days. For the
    consecutive ascending or descending price sequence, display 6th to 9th day
    value.

Args:
    close (pd.Series): Series of 'close's
    asint (bool): If True, fillnas with 0 and change type to int. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    show_all (bool): Show 1 - 13. If set to False, show 6 - 9. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)

Returns:
    pd.DataFrame: New feature generated.
"""
