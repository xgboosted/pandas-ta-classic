"""Numeric parameters reject invalid values instead of substituting a default.

Before 0.9.0 roughly 290 guards of the form
``length = int(length) if length and length > 0 else 10`` silently replaced 0,
negative, NaN, bool and fractional values with the default, so ``sma(length=0)``
returned ``SMA_10``. Every such guard now goes through ``_pos_int`` /
``_pos_float``: ``None`` still selects the default, anything invalid raises
``ValueError`` naming the indicator, the parameter and the value.
"""

import inspect
import re

import numpy as np
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.utils._core import _pos_float, _pos_int
from tests.config import get_sample_data

_GUARD = re.compile(r"_pos_(?:int|float)\((\w+),")
_SERIES = ("open_", "high", "low", "close", "volume")


def _required_extras(name, frame):
    """Inputs that are not price columns but have no default."""
    if name in ("long_run", "short_run"):
        return {"fast": frame.close.rolling(5).mean(), "slow": frame.close.rolling(20).mean()}
    if name == "mavp":
        return {"periods": frame.close * 0 + 10}
    return {}


def _guarded_params():
    cases = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        params = inspect.signature(func).parameters
        guarded = set(_GUARD.findall(inspect.getsource(inspect.getmodule(inspect.unwrap(func)))))
        cases += [(name, p) for p in sorted(guarded & set(params))]
    return cases


CASES = _guarded_params()


def test_sweep_found_the_guards():
    assert len(CASES) > 250


@pytest.fixture(scope="module")
def frame():
    return get_sample_data().iloc[:300]


@pytest.mark.parametrize(("name", "param"), CASES)
def test_negative_value_raises_value_error(name, param, frame):
    func = _find_indicator_func(name)
    kwargs = {p: frame[p.rstrip("_")] for p in inspect.signature(func).parameters if p in _SERIES}
    with pytest.raises(ValueError, match=rf"\b{param} must be an? (integer|number)"):
        func(**kwargs, **_required_extras(name, frame), **{param: -1})


@pytest.mark.parametrize("bad", [0, -5, 2.7, True, np.nan, float("inf"), "14"])
def test_pos_int_rejects(bad):
    def sma(length=None):
        return _pos_int(length, 10, "length")

    with pytest.raises(ValueError, match=r"sma\(\) length must be an integer > 0"):
        sma(bad)


def test_pos_helpers_accept_none_and_valid_values():
    assert _pos_int(None, 10, "length") == 10
    assert _pos_int(14, 10, "length") == 14
    assert _pos_int(14.0, 10, "length") == 14
    assert _pos_int(np.int64(5), 10, "length") == 5
    assert _pos_float(0.5, 0.2, "na", lt=1) == 0.5
    assert _pos_int(0, 1, "ddof", gt=None, ge=0, lt=30) == 0


def test_direct_call_message_names_indicator_and_value():
    close = get_sample_data().close
    with pytest.raises(ValueError, match=r"^sma\(\) length must be an integer > 0, got 0$"):
        ta.sma(close, length=0)
    with pytest.raises(ValueError, match=r"^ebsw\(\) length must be an integer > 38, got 10$"):
        ta.ebsw(close, length=10)
