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
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_float, _pos_int, _str_param
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


def test_number_accepts_zero_and_negatives_but_not_junk():
    assert _number(None, 100, "scalar") == 100
    assert _number(0, 100, "scalar") == 0.0  # used to become 100
    assert _number(-2, 100, "scalar") == -2.0
    for bad in (np.nan, float("inf"), True, "100"):
        with pytest.raises(ValueError, match="scalar must be a number"):
            _number(bad, 100, "scalar")


def test_scalar_zero_is_honoured_and_drift_offset_are_validated():
    close = get_sample_data().close.iloc[:200]
    assert (ta.rsi(close, scalar=0).dropna() == 0).all()  # used to be ta.rsi(close) * 1
    with pytest.raises(ValueError, match=r"^rsi\(\) drift must be an integer > 0, got 0$"):
        ta.rsi(close, drift=0)
    with pytest.raises(ValueError, match=r"^rsi\(\) offset must be an integer, got 2.5$"):
        ta.rsi(close, offset=2.5)
    assert ta.rsi(close, offset=-1).equals(ta.rsi(close).shift(-1))  # negative offsets stay allowed


def test_bool_param_rejects_non_bools():
    def ema(talib=None):
        return _bool_param(talib, False, "talib")

    assert ema() is False
    assert ema(True) is True
    assert ema(np.bool_(True)) is True
    for bad in (1, 0, "True", []):  # talib=1 used to mean False
        with pytest.raises(ValueError, match=r"ema\(\) talib must be True or False"):
            ema(bad)


def test_str_param_rejects_non_strings_and_unknown_choices():
    def cpr(method=None):
        return _str_param(method, "classic", "method", choices={"classic", "woodie"})

    assert cpr() == "classic"
    assert cpr("WOODIE") == "woodie"
    for bad in (1, "", "fibonaci"):
        with pytest.raises(ValueError, match=r"cpr\(\) method must be"):
            cpr(bad)


def test_behaviour_fixes_from_the_same_sweep():
    close = get_sample_data().close.iloc[:200]
    # slope: as_angle=bool(isinstance(as_angle, bool)) made as_angle=False return angles
    assert ta.slope(close, as_angle=False).equals(ta.slope(close))
    assert not ta.slope(close, as_angle=True).equals(ta.slope(close))
    # ma: an unknown name used to return an EMA
    with pytest.raises(ValueError, match=r"ma\(\) name must be one of"):
        ta.ma("emaa", close)
    assert ta.ma(None, close).equals(ta.ema(close))
    # mamode is validated through ma()
    with pytest.raises(ValueError, match=r"mamode must be a non-empty string"):
        ta.apo(close, mamode=123)
    with pytest.raises(ValueError, match=r"ma\(\) name must be one of"):
        ta.apo(close, mamode="foo")
    # tos_stdevall: a bad stds used to become [1, 2, 3]
    with pytest.raises(ValueError, match=r"stds must be a non-empty list"):
        ta.tos_stdevall(close, stds=2)
    # decay: mode="exponential" used to run the linear decay
    assert ta.decay(close > close.shift(), mode="exponential").name.startswith("EXPDECAY")

