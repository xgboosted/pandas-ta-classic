from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._njit import njit

# TD Sequential caps the consecutive run at the 13-bar setup/countdown window.
_TD_WINDOW = 13


@njit(cache=True)
def _td_run_capped(td_bool: np.ndarray) -> np.ndarray:
    """Length of the consecutive-True run ending at each bar, capped at 13.

    Equivalent to the original ``rolling(13).apply(true_sequence_count)``:
    the run resets on every False and never exceeds the 13-bar window.
    """
    n = td_bool.shape[0]
    out = np.zeros(n, dtype=np.float64)
    count = 0
    for i in range(n):
        count = count + 1 if td_bool[i] else 0
        out[i] = count if count < _TD_WINDOW else _TD_WINDOW
    return out


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
        td_bool = series.diff(4) > 0 if direction == "up" else series.diff(4) < 0
        td_num = Series(_td_run_capped(td_bool.to_numpy()))

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
    up_seq, down_seq = apply_offset([up_seq, down_seq], offset)

    # Handle fills
    up_seq, down_seq = apply_fill([up_seq, down_seq], **kwargs)

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
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: New feature generated.
"""
