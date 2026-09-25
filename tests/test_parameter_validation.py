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
    if name == "tsignals":
        return {"trend": (frame.close > frame.close.shift()).astype(int)}
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


def test_slope_has_no_vertical_parameter():
    """vertical was accepted and never used; it is gone from the signature."""
    assert "vertical" not in inspect.signature(ta.slope).parameters


def test_squeeze_pro_rejects_unordered_scalars():
    """An unordered set of Keltner scalars used to return None silently."""
    frame = get_sample_data().iloc[:300]
    with pytest.raises(ValueError, match=r"kc_scalar_wide > kc_scalar_normal > kc_scalar_narrow"):
        ta.squeeze_pro(frame.high, frame.low, frame.close, kc_scalar_wide=1, kc_scalar_normal=1.5, kc_scalar_narrow=2)



# Guards the numeric sweep's grep missed: is_percent(), membership tests,
# "0 < x < 1" ranges, bool(x) coercion, int(kwargs[...]) and abs(n).
_F = get_sample_data().iloc[:300]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: ta.cdl_doji(_F.open, _F.high, _F.low, _F.close, factor=-1), r"cdl_doji\(\) factor must be a number >= 0, got -1"),
        (lambda: ta.increasing(_F.close, percent=-5), r"increasing\(\) percent must be a number >= 0"),
        (lambda: ta.decreasing(_F.close, percent="5"), r"decreasing\(\) percent must be a number"),
        (lambda: ta.ssf(_F.close, poles=4), r"ssf\(\) poles must be 2 or 3, got 4"),
        (lambda: ta.mcgd(_F.close, c=1.5), r"mcgd\(\) c must be a number > 0 and <= 1, got 1.5"),
        (lambda: ta.mcgd(_F.close, c=0), r"mcgd\(\) c must be a number > 0"),
        (lambda: ta.lrsi(_F.close, gamma=1), r"lrsi\(\) gamma must be a number > 0 and < 1, got 1"),
        (lambda: ta.tsignals(_F.close > _F.open, trade_offset=1.5), r"tsignals\(\) trade_offset must be an integer >= 0"),
        (lambda: ta.jma(_F.close, phase=np.nan), r"jma\(\) phase must be a number"),
        (lambda: ta.rainbow(_F.close, num_ribbons=2.5), r"rainbow\(\) num_ribbons must be an integer > 0"),
        (lambda: ta.cdl_z(_F.open, _F.high, _F.low, _F.close, full="no"), r"cdl_z\(\) full must be True or False"),
        (lambda: ta.hwc(_F.close, channel_eval=1), r"hwc\(\) channel_eval must be True or False"),
        (lambda: ta.log_return(_F.close, cumulative="False"), r"log_return\(\) cumulative must be True or False"),
        (lambda: ta.percent_return(_F.close, cumulative=0), r"percent_return\(\) cumulative must be True or False"),
        (lambda: ta.wma(_F.close, asc="False"), r"wma\(\) asc must be True or False"),
        (lambda: ta.fwma(_F.close, asc=0), r"fwma\(\) asc must be True or False"),
        (lambda: ta.sma(_F.close, min_periods=2.7), r"sma\(\) min_periods must be an integer >= 0, got 2.7"),
        (lambda: ta.donchian(_F.high, _F.low, upper_min_periods=1.5), r"donchian\(\) upper_min_periods must be an integer >= 0"),
        (lambda: ta.utils.fibonacci(n=-1), r"fibonacci\(\) n must be an integer >= 0, got -1"),
        (lambda: ta.utils.symmetric_triangle(n=-4), r"symmetric_triangle\(\) n must be an integer > 0, got -4"),
        (lambda: ta.utils.symmetric_triangle(n=0), r"symmetric_triangle\(\) n must be an integer > 0, got 0"),
        (lambda: ta.utils.combination(n=-5, r=2), r"combination\(\) n must be an integer >= 0, got -5"),
    ],
)
def test_remaining_silent_defaults_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_values_the_old_guards_replaced_are_now_honoured():
    doji = ta.cdl_doji(_F.open, _F.high, _F.low, _F.close, factor=50)
    assert doji.name == "CDL_DOJI_10_0.5"  # 50 was accepted before too; 150 became 10
    assert ta.cdl_doji(_F.open, _F.high, _F.low, _F.close, factor=150).name == "CDL_DOJI_10_1.5"
    assert ta.increasing(_F.close, percent=150).name == "INCp_1_1.5"  # 150 used to switch percent mode off
    assert ta.sma(_F.close, min_periods=0).notna().all()
    assert ta.jma(_F.close, phase=0.0).name == "JMA_7_0"


def test_strategy_params_must_be_a_tuple():
    """A list or scalar params was replaced by () and the indicator ran with its defaults."""
    frame = _F.copy()
    frame.ta.cores = 0
    with pytest.raises(TypeError, match=r"Strategy entry 'ema': params must be a tuple, got list \[5\]"):
        frame.ta.strategy(ta.Strategy(name="p", ta=[{"kind": "ema", "params": [5]}]))
    frame.ta.strategy(ta.Strategy(name="p", ta=[{"kind": "ema", "params": (5,)}]))
    assert "EMA_5" in frame.columns


# Numeric options read from **kwargs, and candle penetration, were used unvalidated:
# emv(divisor=0) returned all -0.0, mmar(num_ribbons=-1) an empty frame, rsi(xa="x")
# silently dropped its RSI_14_A column, and a negative penetration produced signals
# where TA-Lib reports TA_BAD_PARAM.
@pytest.mark.parametrize(
    ("call", "message"),
    [
        *[
            (lambda n=n: ta.cdl_pattern(_F.open, _F.high, _F.low, _F.close, name=n, penetration=-1), rf"cdl_{n}\(\) penetration must be a number >= 0, got -1")
            for n in ("eveningstar", "morningstar", "darkcloudcover", "mathold", "abandonedbaby", "eveningdojistar", "morningdojistar")
        ],
        (lambda: ta.emv(_F.high, _F.low, _F.volume, divisor=0), r"emv\(\) divisor must be a number > 0, got 0"),
        (lambda: ta.mmar(_F.close, step=0), r"mmar\(\) step must be an integer > 0, got 0"),
        (lambda: ta.mmar(_F.close, num_ribbons=-1), r"mmar\(\) num_ribbons must be an integer > 0, got -1"),
        (lambda: ta.cpr(_F.open, _F.high, _F.low, _F.close, width_narrow=-1), r"cpr\(\) width_narrow must be a number >= 0, got -1"),
        (lambda: ta.cpr(_F.open, _F.high, _F.low, _F.close, virgin_cpr=True, virgin_lookforward=0), r"cpr\(\) virgin_lookforward must be an integer > 0, got 0"),
        (lambda: ta.aobv(_F.close, _F.volume, run_length=0), r"aobv\(\) run_length must be an integer > 0, got 0"),
        (lambda: ta.rsi(_F.close, signal_indicators=True, xa="x"), r"rsi\(\) xa must be a number, got 'x'"),
        (lambda: ta.rsx(_F.close, signal_indicators=True, xb=True), r"rsx\(\) xb must be a number, got True"),
        (lambda: ta.er(_F.close, signal_indicators=True, xa=np.nan), r"er\(\) xa must be a number, got nan"),
        (lambda: ta.macd(_F.close, signal_indicators=True, xa="0"), r"macd\(\) xa must be a number, got '0'"),
    ],
)
def test_unvalidated_kwargs_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_signal_thresholds_keep_their_column_names():
    """Validation must not turn the caller's 70 into 70.0 (RSI_14_A_70_0); numpy integers are accepted."""
    assert list(ta.rsi(_F.close, signal_indicators=True, xa=70).columns) == ["RSI_14", "RSI_14_A_70", "RSI_14_B_20"]
    assert list(ta.rsi(_F.close, signal_indicators=True, xa=np.int64(70)).columns) == ["RSI_14", "RSI_14_A_70", "RSI_14_B_20"]


# True/False options read from **kwargs were used by truthiness, so detailed="no"
# meant True and lookahead=0 meant False. Each must be True or False now.
# Match converted flags and raw True/False defaults alike, so a site that loses its
# _bool_param wrapper stays in the sweep and fails instead of dropping out of it.
_BOOL_KWARG = re.compile(r'_bool_param\(kwargs\.(?:pop|get)\("(\w+)"|kwargs\.(?:pop|get)\("(\w+)", (?:True|False)\)')
_NOT_FLAGS = {"ma1", "ma2", "osc", "tulipy"}  # stc's optional Series; msw's backend switch
_SIGNAL_GATED = {"cross_values", "cross_series"}


def _bool_kwarg_cases():
    cases = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        found = _BOOL_KWARG.findall(inspect.getsource(inspect.getmodule(inspect.unwrap(func))))
        keys = sorted({a or b for a, b in found} - _NOT_FLAGS)
        cases += [(name, key) for key in keys if name != "macd"]
    return cases


BOOL_KWARG_CASES = _bool_kwarg_cases()


def test_bool_kwarg_sweep_found_the_flags():
    assert len(BOOL_KWARG_CASES) >= 30


@pytest.mark.parametrize(("name", "key"), BOOL_KWARG_CASES)
def test_bool_kwarg_rejects_non_bool(name, key, frame):
    func = _find_indicator_func(name)
    kwargs = {p: frame[p.rstrip("_")] for p in inspect.signature(func).parameters if p in _SERIES}
    extra = {"signal_indicators": True} if key in _SIGNAL_GATED else {}
    with pytest.raises(ValueError, match=rf"{name}\(\) {key} must be True or False, got 'no'"):
        func(**kwargs, **_required_extras(name, frame), **extra, **{key: "no"})


def test_strategy_and_metric_flags_reject_non_bool(frame):
    df = frame.copy()
    df.ta.cores = 0
    for key in ("verbose", "timed", "ordered", "returns"):
        with pytest.raises(ValueError, match=rf"strategy\(\) {key} must be True or False"):
            df.ta.strategy("candles", **{key: 1})
    with pytest.raises(ValueError, match=r"volatility\(\) nearest_day must be True or False"):
        ta.utils.volatility(frame.close, nearest_day="yes")


def test_vwap_without_datetime_index_raises_type_error(frame):
    """vwap anchors by calendar period; a RangeIndex used to fail with an unrelated AttributeError."""
    flat = frame.reset_index(drop=True)
    with pytest.raises(TypeError, match=r"vwap\(\) needs a DatetimeIndex to anchor by 'D', got RangeIndex"):
        ta.vwap(flat.high, flat.low, flat.close, flat.volume)


def test_strategy_skips_vwap_without_datetime_index(frame):
    """strategy("all") on a RangeIndex frame crashed on vwap; all/category modes now leave it out."""
    flat = frame.reset_index(drop=True).copy()
    flat.ta.cores = 0
    flat.ta.strategy("overlap")
    assert "SMA_10" in flat.columns
    assert not any(c.startswith("VWAP") for c in flat.columns)


def test_trend_reset_is_removed(frame):
    """trend_reset was documented as ending a trend but never read (AGENTS rule 4, rule 11 exception)."""
    trend = (frame.close > frame.open).astype(int)
    for call, name in ((lambda: ta.tsignals(trend, trend_reset=1), "tsignals"), (lambda: ta.xsignals(frame.close, 50, 40, trend_reset=0), "xsignals")):
        with pytest.raises(TypeError, match=rf"{name}\(\) no longer accepts 'trend_reset'"):
            call()
    # trade_offset is keyword-only, so an old positional trend_reset cannot slide into it
    with pytest.raises(TypeError):
        ta.tsignals(trend, False, 1)


# Boolean parameters declared with a True/False default in the signature were read by
# truthiness too (xsignals(above=0) meant below, ichimoku(include_chikou=0) dropped the
# Chikou column). Cases come from the signatures, not from the validation calls, so a
# parameter that loses its _bool_param call still fails here.
def _signature_bool_cases():
    cases = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        for param in inspect.signature(func).parameters.values():
            if isinstance(param.default, bool):
                cases.append((name, param.name))
    return cases


SIGNATURE_BOOL_CASES = _signature_bool_cases()


def test_signature_bool_sweep_found_the_parameters():
    assert len(SIGNATURE_BOOL_CASES) >= 10  # 11 before ichimoku's as_dataframe was removed


@pytest.mark.parametrize(("name", "param"), SIGNATURE_BOOL_CASES)
def test_signature_bool_rejects_non_bool(name, param, frame):
    func = _find_indicator_func(name)
    kwargs = {p: frame[p.rstrip("_")] for p in inspect.signature(func).parameters if p in _SERIES}
    extra = {"xa": 50, "xb": 40} if name == "xsignals" else {}
    if name == "xsignals":
        kwargs = {"signal": frame.close}
    with pytest.raises(ValueError, match=rf"{name}\(\) {param} must be True or False, got 0"):
        func(**kwargs, **_required_extras(name, frame), **extra, **{param: 0})


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: ta.utils.combination(n=5, r=2, repetition=1), r"combination\(\) repetition must be True or False"),
        (lambda: ta.utils.combination(n=5, r=2, multichoose="yes"), r"combination\(\) multichoose must be True or False"),
        (lambda: ta.utils.fibonacci(n=5, zero=1), r"fibonacci\(\) zero must be True or False"),
        (lambda: ta.utils.fibonacci(n=5, weighted="no"), r"fibonacci\(\) weighted must be True or False"),
        (lambda: ta.utils.pascals_triangle(n=4, inverse=1), r"pascals_triangle\(\) inverse must be True or False"),
        (lambda: ta.utils.symmetric_triangle(n=4, weighted=0), r"symmetric_triangle\(\) weighted must be True or False"),
        (lambda: ta.utils.unsigned_differences(_F.close, asint="x"), r"unsigned_differences\(\) asint must be True or False"),
    ],
)
def test_helper_bool_options_reject_non_bool(call, message):
    with pytest.raises(ValueError, match=message):
        call()
