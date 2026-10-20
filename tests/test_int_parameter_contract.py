"""A small whole-number parameter must not blank a column on long input.

`kama(drift=2)` published nothing at all on 1000 bars: its efficiency
ratio needs `length` price changes `drift` bars apart, the first of which
does not exist until bar `drift`, and the recursion started at bar
`length` from a NaN ratio, so the NaN rode it to the end. Every int
parameter of every registered indicator is set to 2, 3 and 5; a column the
default call fills may not come back with at most one value.
"""

import inspect
import typing
import warnings

import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, NOT_OHLCV, inputs, params, spy

VALUES = (2, 3, 5)
CASES = [
    (name, param)
    for name in INDICATORS
    if name not in NOT_OHLCV
    for param, spec in inspect.signature(getattr(ta, name)).parameters.items()
    if param != "offset" and (spec.annotation is int or int in typing.get_args(spec.annotation))
]
KNOWN = {
    ("vidya", "drift"): "OPEN: the CMO starts at bar length + drift - 1, the SMA seed sits at bar length - 1; where to seed is undecided",
}


@pytest.fixture(scope="module")
def frame():
    return spy(1000)


def _frame(result) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


def test_sweep_found_the_parameters():
    assert ("kama", "drift") in CASES and len(CASES) > 150


@pytest.mark.parametrize(("name", "param"), params(CASES, KNOWN, id=lambda row: "-".join(row)))
def test_a_small_int_parameter_leaves_values(name, param, frame):
    func = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _frame(func(**inputs(name, frame)))
        blank = []
        for value in VALUES:
            try:
                result = _frame(func(**inputs(name, frame), **{param: value}))
            except ValueError:  # outside the parameter's range (fast >= slow, ddof >= length, ...)
                continue
            blank += [f"{param}={value}: {c}" for c in result.columns if c in default and default[c].notna().any() and result[c].notna().sum() <= 1]
    assert not blank, f"{name}: {blank}"
