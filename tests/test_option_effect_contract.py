"""An option the caller sets changes the result (AGENTS rules 1 and 5).

`squeeze(mamode="wma")` smoothed its momentum with an SMA anyway (only "ema"
was read, everything else fell through to SMA); `qstick(ma="wma")` did the
same; `squeeze(use_tr=False)` was declared but never read. Each call looked
like it worked. Two registry sweeps catch that class:

* every MA choice an indicator takes (`mamode`, or `ma` read from **kwargs)
  changes every column it smooths: sma, ema, wma and rma never coincide for a
  length >= 2, so two of them giving the same column means the choice was
  dropped somewhere. Columns the choice does not reach by definition are listed
  in ``INDEPENDENT_OF_MA`` with their reason.
* every declared boolean parameter changes the values or the names when flipped.
"""

import inspect
import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, NOT_OHLCV, boolean_kwargs, inputs, params, spy

MAS = ("sma", "ema", "wma", "rma")

# (indicator, column of its default call) -> why no MA choice reaches it.
INDEPENDENT_OF_MA = {
    ("aobv", "OBV"): "KEPT: the raw OBV; mamode smooths its fast/slow lines",
    ("aobv", "OBV_min_2"): "KEPT: rolling minimum of the raw OBV",
    ("aobv", "OBV_max_2"): "KEPT: rolling maximum of the raw OBV",
    ("stochf", "STOCHFk_5_3"): "KEPT: the raw fast %K; mamode smooths %D",
    ("thermo", "THERMO_20_2_0.5"): "KEPT: the raw thermometer; mamode smooths its signal line",
    ("tsi", "TSI_13_25_13"): "KEPT: the TSI line is double EMA by definition; mamode smooths the signal",
    ("squeeze", "SQZ_NO"): "KEPT: with bb_length == kc_length both bands are symmetric around one midline, so 'no squeeze' is 0 on every bar; only its warm-up follows the MA (dema, t3 start later)",
    ("squeeze_pro", "SQZPRO_NO"): "KEPT: with bb_length == kc_length both bands are symmetric around one midline, so 'no squeeze' is 0 on every bar; only its warm-up follows the MA (dema, t3 start later)",
}
# indicator -> boolean parameter that only acts together with another one.
DEPENDENT_FLAGS = {
    ("slope", "to_degrees"): "KEPT: only read when as_angle=True",
}


def _ma_options(name: str) -> list[str]:
    func = getattr(ta, name)
    declared = ["mamode"] if "mamode" in inspect.signature(func).parameters else []
    source = inspect.getsource(inspect.getmodule(inspect.unwrap(func)))
    return declared + sorted(set(re.findall(r'kwargs\.(?:get|pop|setdefault)\("(ma|mamode)"', source)) - set(declared))


MA_CASES = [(name, option) for name in INDICATORS if name not in NOT_OHLCV for option in _ma_options(name)]
BOOL_CASES = [(name, flag) for name in INDICATORS if name not in NOT_OHLCV for flag in boolean_kwargs(name)]


def _frame(result) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


@pytest.fixture(scope="module")
def frame():
    return spy()


@pytest.mark.parametrize(("name", "option"), MA_CASES, ids=[f"{n}-{o}" for n, o in MA_CASES])
def test_every_ma_choice_reaches_every_smoothed_column(name, option, frame):
    func = getattr(ta, name)
    results = {ma: _frame(func(**inputs(name, frame), **{option: ma})) for ma in MAS}
    default_columns = list(_frame(func(**inputs(name, frame))).columns)
    unchanged = set()
    for a, b in combinations(MAS, 2):
        x, y = results[a], results[b]
        for i, column in enumerate(default_columns):
            if np.array_equal(x.iloc[:, i].to_numpy(float, na_value=np.nan), y.iloc[:, i].to_numpy(float, na_value=np.nan), equal_nan=True):
                unchanged.add(column)
    dropped = sorted(c for c in unchanged if (name, c) not in INDEPENDENT_OF_MA)
    assert not dropped, f"{name}({option}=...) gives the same {dropped} for two of {MAS}"


@pytest.mark.parametrize(("name", "column"), sorted(INDEPENDENT_OF_MA), ids=[f"{n}-{c}" for n, c in sorted(INDEPENDENT_OF_MA)])
def test_independent_columns_are_independent(name, column, frame):
    """A KEPT row is checked too: its column must exist and be the same for every MA choice."""
    func = getattr(ta, name)
    results = [_frame(func(**inputs(name, frame), mamode=ma))[column].to_numpy(float, na_value=np.nan) for ma in MAS]
    assert all(np.array_equal(results[0], other, equal_nan=True) for other in results[1:])


def test_qstick_forwards_the_options_of_its_ma(frame):
    """qstick(ma="ema", adjust=True) handed adjust to ema() before qstick went through ma();
    the fill and the offset stay with qstick, so each applies once."""
    diff = frame.close - frame.open
    expected = ta.ema(diff, length=10, adjust=True)
    assert np.allclose(ta.qstick(frame.open, frame.close, ma="ema", adjust=True), expected, equal_nan=True)
    shifted = ta.ema(diff, length=10).shift(2).fillna(0)
    assert np.array_equal(ta.qstick(frame.open, frame.close, ma="ema", offset=2, fillna=0), shifted)


@pytest.mark.parametrize(("name", "flag"), params(BOOL_CASES, DEPENDENT_FLAGS, id=lambda row: "-".join(row)))
def test_every_boolean_parameter_changes_the_result(name, flag, frame):
    func = getattr(ta, name)
    on = _frame(func(**inputs(name, frame), **{flag: True}))
    off = _frame(func(**inputs(name, frame), **{flag: False}))
    same_values = on.shape == off.shape and np.array_equal(on.to_numpy(float, na_value=np.nan), off.to_numpy(float, na_value=np.nan), equal_nan=True)
    assert not (same_values and list(on.columns) == list(off.columns)), f"{name}({flag}=True) == {name}({flag}=False)"


@pytest.mark.parametrize("name", ["squeeze", "squeeze_pro"])
def test_squeeze_tr_is_the_older_name_of_use_tr(name, frame):
    """The docstring listed `tr` under Kwargs and the signature `use_tr`; only `tr` was read."""
    func = getattr(ta, name)
    df = frame.iloc[-300:]
    assert func(df.high, df.low, df.close, tr=False).equals(func(df.high, df.low, df.close, use_tr=False))
    with pytest.raises(TypeError, match=rf"{name}\(\) got both use_tr and tr"):
        func(df.high, df.low, df.close, use_tr=False, tr=True)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda df: ta.qstick(df.open, df.close, ma="bogus"), r"qstick\(\) ma must be one of"),
        (lambda df: ta.squeeze(df.high, df.low, df.close, mamode="bogus"), r"squeeze\(\) mamode must be one of"),
        (lambda df: ta.squeeze_pro(df.high, df.low, df.close, mamode="bogus"), r"squeeze_pro\(\) mamode must be one of"),
    ],
)
def test_an_unknown_ma_names_the_function_called(call, message, frame):
    """qstick(ma="bogus") used to return an SMA; squeeze's error named bbands' ma()."""
    with pytest.raises(ValueError, match=message):
        call(frame.iloc[-300:])


@pytest.mark.parametrize("option", [{"adjust": True}, {"sma": False}])
def test_ema_options_reach_the_ema_of_dema_kc_apo_and_ppo(option, frame):
    """dema, kc, apo and ppo accepted ema's adjust and sma and returned the default
    result; they now hand them on to their EMAs, as tema and qstick do (#340)."""
    close = frame.close
    e12, e26 = ta.ema(close, length=12, **option), ta.ema(close, length=26, **option)
    assert np.allclose(ta.apo(close, mamode="ema", **option), e12 - e26, equal_nan=True)
    assert np.allclose(ta.ppo(close, mamode="ema", **option).iloc[:, 0], 100 * (e12 - e26) / e26, equal_nan=True)
    assert np.allclose(ta.kc(frame.high, frame.low, close, **option).iloc[:, 1], ta.ema(close, length=20, **option), equal_nan=True)
    assert not np.allclose(ta.dema(close, **option), ta.dema(close), equal_nan=True)
