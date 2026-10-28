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
from pandas import Series, Timedelta

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_float, _pos_int, _str_param
from pandas_ta_classic.utils._signals import attach_signals
from tests.config import get_sample_data
from tests.test_degenerate_input import _call
from tests.test_docstring_defaults import _MODULES, _code_defaults

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


def test_a_whole_number_keeps_the_type_of_an_int_default():
    assert type(_pos_float(2.0, 2, "scalar")) is int
    assert type(_pos_float(2, 2.0, "std")) is float
    assert type(_pos_float(2.5, 2, "scalar")) is float
    assert type(_number(100.0, 100, "scalar")) is int


def _explicit_default_cases():
    cases = []
    for module in _MODULES:
        if module.stem not in _REGISTERED:
            continue
        for param, default in _code_defaults(module.read_text(encoding="utf-8")).items():
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                spellings = [int(default), float(default)] if float(default).is_integer() else [default]
                cases += [(module.stem, param, value) for value in spellings]
    return cases


_REGISTERED = {n for names in ta.Category.values() for n in names}
_EXPLICIT_DEFAULTS = _explicit_default_cases()


def _output_names(result):
    return [result.name] if isinstance(result, Series) else list(result.columns)


@pytest.mark.parametrize(("name", "param", "value"), _EXPLICIT_DEFAULTS)
def test_passing_the_default_explicitly_keeps_the_output_names(name, param, value, frame):
    """``kc(scalar=2)`` named its columns ``KCLe_20_2.0`` where ``kc()`` gives ``KCLe_20_2``.

    ``_pos_float`` returned ``float(2)`` for an int default of 2; chop, eom, kc,
    squeeze_pro and thermo renamed their output when given their own default.
    """
    columns = {"open": frame.open, "high": frame.high, "low": frame.low, "close": frame.close, "volume": frame.volume}
    columns |= {"open_": frame.open, "benchmark": frame.close, "series_a": frame.close, "series_b": frame.open, "source": frame.close}
    default = _call(name, columns)
    if default is None:
        pytest.skip(f"{name} returns None for this input")
    assert _output_names(_call(name, columns, **{param: value})) == _output_names(default)


def test_explicit_default_sweep_covers_the_int_defaults_read_as_floats():
    swept = {(name, param) for name, param, _ in _EXPLICIT_DEFAULTS}
    assert {("chop", "scalar"), ("eom", "divisor"), ("kc", "scalar"), ("thermo", "long"), ("squeeze_pro", "kc_scalar_wide")} <= swept
    assert len(swept) > 300


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
            (
                lambda n=n: ta.cdl_pattern(_F.open, _F.high, _F.low, _F.close, name=n, penetration=-1),
                rf"cdl_{n}\(\) penetration must be a number >= 0, got -1",
            )
            for n in ("eveningstar", "morningstar", "darkcloudcover", "mathold", "abandonedbaby", "eveningdojistar", "morningdojistar")
        ],
        (lambda: ta.emv(_F.high, _F.low, _F.volume, divisor=0), r"emv\(\) divisor must be a number > 0, got 0"),
        (lambda: ta.mmar(_F.close, step=0), r"mmar\(\) step must be an integer > 0, got 0"),
        (lambda: ta.mmar(_F.close, num_ribbons=-1), r"mmar\(\) num_ribbons must be an integer > 0, got -1"),
        (lambda: ta.cpr(_F.open, _F.high, _F.low, _F.close, width_narrow=-1), r"cpr\(\) width_narrow must be a number >= 0, got -1"),
        (
            lambda: ta.cpr(_F.open, _F.high, _F.low, _F.close, virgin_cpr=True, virgin_lookforward=0),
            r"cpr\(\) virgin_lookforward must be an integer > 0, got 0",
        ),
        (lambda: ta.aobv(_F.close, _F.volume, run_length=0), r"aobv\(\) run_length must be an integer > 0, got 0"),
        (lambda: ta.rsi(_F.close, signal_indicators=True, xa="x"), r"rsi\(\) xa must be a number, got 'x'"),
        (lambda: ta.rsx(_F.close, signal_indicators=True, xb=True), r"rsx\(\) xb must be a number, got True"),
        (lambda: ta.er(_F.close, signal_indicators=True, xa=np.nan), r"er\(\) xa must be a number, got nan"),
        (lambda: ta.macd(_F.close, signal_indicators=True, xa="0"), r"macd\(\) xa must be a number, got '0'"),
        # A bad signal option is a caller error even when the signals are off.
        (lambda: ta.macd(_F.close, xa="0"), r"macd\(\) xa must be a number, got '0'"),
        (lambda: ta.macd(_F.close, xb=np.inf), r"macd\(\) xb must be a number, got inf"),
        (lambda: ta.macd(_F.close, cross_values="yes"), r"macd\(\) cross_values"),
        (lambda: ta.macd(_F.close, cross_series=1), r"macd\(\) cross_series"),
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


def _flags(source: str) -> set[str]:
    return {a or b for a, b in _BOOL_KWARG.findall(source)}


# er, rsi and rsx hand their flags to attach_signals, so a per-module scrape no
# longer sees them. Read them off the helper, so a flag it gains joins the sweep.
_HELPER_FLAGS = _flags(inspect.getsource(attach_signals))


def _bool_kwarg_cases():
    cases = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        source = inspect.getsource(inspect.getmodule(inspect.unwrap(func)))
        keys = _flags(source) | (_HELPER_FLAGS if "attach_signals(" in source else set())
        cases += [(name, key) for key in sorted(keys - _NOT_FLAGS) if name != "macd"]
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
    for key in ("verbose", "timed", "returns"):
        with pytest.raises(ValueError, match=rf"strategy\(\) {key} must be True or False"):
            df.ta.strategy("candles", **{key: 1})
    with pytest.raises(ValueError, match=r"volatility\(\) nearest_day must be True or False"):
        ta.utils.volatility(frame.close, nearest_day="yes")


def _fill_inputs(name, func, frame):
    """Every input *name* needs, including the math operators' two operands and xsignals' thresholds."""
    params = inspect.signature(func).parameters
    kwargs = {p: frame[p.rstrip("_")] for p in params if p in _SERIES}
    if "series_a" in params:
        kwargs |= {"series_a": frame.close, "series_b": frame.open}
    if "benchmark" in params:
        kwargs["benchmark"] = frame.open
    if name == "xsignals":
        kwargs |= {"signal": ta.rsi(frame.close), "xa": 70, "xb": 30}
    return kwargs | _required_extras(name, frame)


FILL_NAMES = sorted(i for v in ta.Category.values() for i in v if _find_indicator_func(i) is not None)


@pytest.mark.parametrize("name", FILL_NAMES)
def test_bad_fill_method_names_the_indicator_called(name, frame):
    """apply_fill validates fill_method, but the error named apply_fill(), or the
    inner indicator that applied it (ema for tema, linreg for linregslope)."""
    func = _find_indicator_func(name)
    source = {"name": "sma", "source": frame.close} if name == "ma" else _fill_inputs(name, func, frame)
    with pytest.raises(ValueError, match=rf"^{name}\(\) fill_method must be one of \['bfill', 'ffill'\], got 'bogus'$"):
        func(**source, fill_method="bogus")


def test_bad_fill_method_leaves_the_input_unfilled():
    """fill_method is validated before fillna runs: a rejected call must not have
    filled the caller's Series in place on its way to the error (AGENTS rule 19)."""
    series = Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="fill_method"):
        ta.utils.apply_fill(series, fillna=0, fill_method="bogus")
    assert series.isna().tolist() == [False, True, False]


def test_bad_fill_method_names_the_indicator_through_the_accessor(frame):
    df = frame.copy()
    df.ta.cores = 0
    with pytest.raises(ValueError, match=r"^tema\(\) fill_method"):
        df.ta.tema(fill_method="bogus")
    with pytest.raises(ValueError, match=r"^bbands\(\) fill_method"):
        df.ta.strategy(ta.Strategy("fill", [{"kind": "bbands", "fill_method": "bogus"}]))


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
    for call, name in (
        (lambda: ta.tsignals(trend, trend_reset=1), "tsignals"),
        (lambda: ta.xsignals(frame.close, 50, 40, trend_reset=0), "xsignals"),
    ):
        with pytest.raises(TypeError, match=rf"{name}\(\) no longer accepts 'trend_reset'"):
            call()
    # trade_offset is keyword-only, so an old positional trend_reset cannot slide into it
    with pytest.raises(TypeError):
        ta.tsignals(trend, False, 1)


# Boolean parameters were read by truthiness too (xsignals(above=0) meant below,
# ichimoku(include_chikou=0) dropped the Chikou column). Cases are the signature
# parameters with a True/False default, in the signature or resolved from None by a
# _bool_param call, so a parameter that loses its _bool_param call still fails here.
def _signature_bool_cases():
    cases = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        resolved = _code_defaults(inspect.getsource(inspect.getmodule(inspect.unwrap(func))))
        for param in inspect.signature(func).parameters.values():
            if isinstance(param.default, bool) or (param.default is None and isinstance(resolved.get(param.name), bool)):
                cases.append((name, param.name))
    return cases


SIGNATURE_BOOL_CASES = _signature_bool_cases()


def test_signature_bool_sweep_found_the_parameters():
    assert len(SIGNATURE_BOOL_CASES) >= 90  # 98 once None defaults resolved by _bool_param count too


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


def test_signature_defaults_are_none():
    # None selects the default, resolved by the validation call, so a caller can pass
    # None to mean "the default" everywhere. bbands, cdl_doji, cdl_inside, cdl_pattern,
    # cpr, dpo, ichimoku and xsignals wrote their defaults into the signature instead.
    offenders = []
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        if func is None:
            continue
        for param in inspect.signature(func).parameters.values():
            if param.default is not inspect.Parameter.empty and param.default is not None:
                offenders.append(f"{name}({param.name}={param.default!r})")
    assert not offenders, offenders


# Series options read from **kwargs (the signal helpers' xserie, stc's ma1/ma2/osc)
# with another index than close failed inside pandas ("Can only compare
# identically-labeled Series objects", "Length of values (100) does not match length
# of index (300)"), a message that names neither the indicator nor the option. The
# options are read off each docstring's Kwargs section; macdfix only names the xserie
# options it forwards to macd() in prose.
_SERIES_KWARG = re.compile(r"^\s+(\w+) \((?:pd\.)?Series", re.MULTILINE)
_STC_OPEN = (
    "OPEN: stc's rule is undecided: a shorter ma1 next to a full ma2 is aligned by label, an osc shorter than the "
    "window reads all-NaN (test_stc_short_series_through_kwargs_is_all_nan), a shifted osc is used bar for bar"
)
_MACDFIX_OPEN = "OPEN: macdfix hands its signal options to macd(), whose errors name macd() (so do xa and cross_series)"
_SERIES_KWARG_OPEN = {"stc": _STC_OPEN, "macdfix": _MACDFIX_OPEN}


def _series_kwarg_cases():
    cases = [("macdfix", "xserie")]
    for name in sorted(i for v in ta.Category.values() for i in v):
        func = _find_indicator_func(name)
        doc = func.__doc__ or ""
        if "Kwargs:" in doc:
            params = inspect.signature(func).parameters
            cases += [(name, key) for key in _SERIES_KWARG.findall(doc.split("Kwargs:", 1)[1]) if key not in params]
    return cases


SERIES_KWARG_CASES = _series_kwarg_cases()


def test_series_kwarg_sweep_found_the_options():
    signals = {(name, key) for name in ("er", "macd", "rsi", "rsx") for key in ("xserie", "xserie_a", "xserie_b")}
    assert set(SERIES_KWARG_CASES) == signals | {("macdfix", "xserie"), ("stc", "ma1"), ("stc", "ma2"), ("stc", "osc")}


def _other_index(frame, rows: str) -> Series:
    if rows == "shorter":
        return frame.close.iloc[200:]
    return frame.close.set_axis(frame.close.index + Timedelta(days=1))  # shares most labels with close, not all


@pytest.mark.parametrize(
    ("name", "key"),
    [
        pytest.param(*case, marks=pytest.mark.xfail(strict=True, raises=(AssertionError, pytest.fail.Exception), reason=_SERIES_KWARG_OPEN[case[0]]))
        if case[0] in _SERIES_KWARG_OPEN
        else case
        for case in SERIES_KWARG_CASES
    ],
)
@pytest.mark.parametrize("rows", ["shorter", "shifted"])
def test_series_kwarg_with_another_index_raises(name, key, rows, frame):
    func = _find_indicator_func(name)
    kwargs = {p: frame[p.rstrip("_")] for p in inspect.signature(func).parameters if p in _SERIES}
    partner = {"ma1": "ma2", "ma2": "ma1"}.get(key)
    extra = {partner: frame.close} if partner else {} if key == "osc" else {"signal_indicators": True}
    other = _other_index(frame, rows)
    with pytest.raises(ValueError, match=rf"^{name}\(\) {key} must have the same index as close: {key} has {len(other)} rows"):
        func(**kwargs, **extra, **{key: other})


@pytest.mark.parametrize(("name", "key"), [case for case in SERIES_KWARG_CASES if case[0] != "stc"])
def test_series_signal_option_is_not_checked_while_the_signals_are_off(name, key, frame):
    """Without signal_indicators the option is not read, as before: nothing to raise for."""
    func = _find_indicator_func(name)
    assert func(frame.close).equals(func(frame.close, **{key: _other_index(frame, "shorter")}))
