"""Unknown keywords raise TypeError, and the table of accepted keywords matches the code (issue #340).

Every indicator took ``**kwargs`` and rejected no name in it, so ``rsi(close,
lenght=7)`` returned RSI_14 without a word, and so did ``df.ta.rsi(lenght=7)``,
a Strategy entry ``{"kind": "rsi", "lenght": 7}`` and ``strategy("momentum",
lenght=7)``. ``pandas_ta_classic/utils/_keywords.py`` now lists what each
indicator accepts, and the outermost call checks it.
"""

import ast
import functools
import inspect
import textwrap
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.candles.cdl_pattern import _NATIVE_PATTERNS, ALL_PATTERNS
from pandas_ta_classic.overlap.ma import _MA_DISPATCH, ma as _ma
from pandas_ta_classic.utils._keywords import _FILL, _KWARGS_OPTIONS, _MA_CHOICE, _MA_OPTIONS, _MA_TALIB, _PATTERN_OPTIONS, choice_keywords
from tests.config import get_sample_data
from tests.contracts import INDICATORS, inputs
from tests.test_docstring_params import _KWARG_READ, _is_kwargs, _resolve


@functools.cache
def _scan(fn) -> tuple[frozenset, bool]:
    """The names a caller can pass in *fn*'s ``**kwargs``, and whether they reach ``ma()``.

    A name counts when *fn* reads it from ``kwargs``, or when *fn* hands
    ``**kwargs`` to a package function whose parameter of that name the call
    does not fill itself (``ema(close, length=fast, **kwargs)`` passes ``talib``
    on, not ``close`` or ``length``).
    """
    fn = inspect.unwrap(fn)
    source = textwrap.dedent(inspect.getsource(fn))
    names = {a or b for a, b in _KWARG_READ.findall(source)}
    via_ma = False
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        star = any(k.arg is None and _is_kwargs(k.value) for k in node.keywords)
        plain = any(k.arg is not None and _is_kwargs(k.value) for k in node.keywords) or any(_is_kwargs(a) for a in node.args)
        if not (star or plain):
            continue
        for target in _resolve(node.func, fn.__globals__):
            target = inspect.unwrap(target) if callable(target) else None
            if target is inspect.unwrap(_ma):
                via_ma = True
                continue
            if not (inspect.isfunction(target) and target.__module__.startswith("pandas_ta_classic") and target is not fn):
                continue
            sub, sub_ma = _scan(target)
            names |= sub
            via_ma |= sub_ma
            if star:
                params = [p for p, s in inspect.signature(target).parameters.items() if s.kind not in (s.VAR_KEYWORD, s.VAR_POSITIONAL)]
                given = set(params[: len(node.args)]) | {k.arg for k in node.keywords if k.arg}
                names |= set(params) - given
    return frozenset(names), via_ma


def _extra(fn) -> frozenset:
    return _scan(fn)[0] - set(inspect.signature(fn).parameters) - _FILL


@pytest.mark.parametrize("name", INDICATORS)
def test_the_table_lists_what_the_code_reads(name):
    func = _find_indicator_func(name)
    if name == "ma":
        # ma() hands every keyword to the moving average it dispatches to
        expected = frozenset().union(*(choice_keywords("ma", {"name": m}) for m in _MA_DISPATCH))
        assert _extra(func) == expected
        return
    if name == "cdl_pattern":
        # the scan sees the native patterns' penetration only; _PATTERN_OPTIONS is checked below
        assert _extra(func) <= frozenset().union(*_PATTERN_OPTIONS.values())
        assert func.__name__ not in _KWARGS_OPTIONS
        return
    assert _extra(func) == _KWARGS_OPTIONS.get(func.__name__, frozenset()), f"{name}: update _KWARGS_OPTIONS"


def test_the_table_names_only_registered_indicators():
    names = {_find_indicator_func(name).__name__ for name in INDICATORS}
    assert set(_KWARGS_OPTIONS) <= names
    assert set(_MA_CHOICE) <= names


def test_every_indicator_that_reaches_ma_is_in_the_choice_table():
    # kc, apo and ppo pass the MA's options in a dict of their own, which the scan does not follow
    reach = {name for name in INDICATORS if name != "ma" and _scan(_find_indicator_func(name))[1]}
    assert reach | {"kc", "apo", "ppo"} == set(_MA_CHOICE)


@pytest.mark.parametrize("name", sorted(_MA_CHOICE))
def test_the_choice_table_has_the_defaults(name):
    parameter, default, _ = _MA_CHOICE[name]
    df = get_sample_data().iloc[-300:]
    func = getattr(ta, name)
    implicit = func(**inputs(name, df))
    explicit = func(**inputs(name, df), **{parameter: default})
    assert np.array_equal(pd.DataFrame(implicit).to_numpy(float), pd.DataFrame(explicit).to_numpy(float), equal_nan=True)
    other = func(**inputs(name, df), **{parameter: "hma"})
    assert not np.array_equal(pd.DataFrame(implicit).to_numpy(float), pd.DataFrame(other).to_numpy(float), equal_nan=True), name


def test_the_ma_tables_match_the_moving_averages():
    talib = {name for name, func in _MA_DISPATCH.items() if "talib" in inspect.signature(func).parameters}
    assert talib == _MA_TALIB
    assert set(_MA_OPTIONS) <= set(_MA_DISPATCH)


def test_the_pattern_table_matches_the_patterns():
    found = {}
    for pattern in ALL_PATTERNS:
        func = {"doji": ta.cdl_doji, "inside": ta.cdl_inside}.get(pattern) or _NATIVE_PATTERNS[pattern]
        own = set(inspect.signature(func).parameters) - {"open_", "high", "low", "close", "scalar", "offset", "kwargs"}
        options = own | _KWARGS_OPTIONS.get(func.__name__, frozenset())
        if options:
            found[pattern] = options
    assert found == _PATTERN_OPTIONS


@pytest.fixture(scope="module")
def df():
    return get_sample_data().iloc[-300:].copy()


@pytest.mark.parametrize("name", INDICATORS)
def test_an_unknown_keyword_raises(name, df):
    with pytest.raises(TypeError, match=rf"^{name if name not in ('max', 'min', 'sum') else 'rolling_' + name}\(\) got an unexpected keyword argument 'lenght'$"):
        getattr(ta, name)(**inputs(name, df), lenght=7)


def test_several_unknown_keywords_are_named_together(df):
    with pytest.raises(TypeError, match=r"^rsi\(\) got unexpected keyword arguments 'lenght', 'ofset'$"):
        ta.rsi(df.close, lenght=7, ofset=1)


def test_the_accessor_raises_and_keeps_its_own_keywords(df):
    with pytest.raises(TypeError, match=r"^rsi\(\) got an unexpected keyword argument 'lenght'$"):
        df.ta.rsi(lenght=7)
    frame = df.copy()
    result = frame.ta.sma(length=5, append=True, prefix="P", suffix="S", delimiter="-", col_names=("MY",), verbose=False)
    assert result.name == "P-SMA_5-S"
    assert "MY" in frame.columns


def test_a_moving_average_option_is_accepted_when_that_average_reads_it(df):
    assert ta.bbands(df.close, mamode="ema", adjust=False) is not None
    assert ta.bbands(df.close, 20, 2.0, 0, "ema", adjust=False) is not None  # the choice by position
    assert ta.ma("zlma", df.close, mamode="sma") is not None
    assert ta.bias(df.close, mamode="sma", talib=False) is not None
    with pytest.raises(TypeError, match=r"^bbands\(\) got an unexpected keyword argument 'adjust'$"):
        ta.bbands(df.close, adjust=False)  # sma
    with pytest.raises(TypeError, match=r"^kc\(\) got an unexpected keyword argument 'talib'$"):
        ta.kc(df.high, df.low, df.close, talib=False)  # kc passes the EMA its options only


def test_a_pattern_option_is_accepted_when_that_pattern_reads_it(df):
    assert ta.cdl_pattern(df.open, df.high, df.low, df.close, name="doji", naive=True) is not None
    assert ta.cdl_pattern(df.open, df.high, df.low, df.close, name="morningstar", penetration=0.5) is not None
    with pytest.raises(TypeError, match=r"^cdl_pattern\(\) got an unexpected keyword argument 'naive'$"):
        ta.cdl_pattern(df.open, df.high, df.low, df.close, name="inside", naive=True)


def test_a_removed_name_keeps_its_own_message(df):
    with pytest.raises(TypeError, match="no longer accepts 'interval'"):
        ta.cpr(df.open, df.high, df.low, df.close, interval=1)


def test_a_strategy_entry_raises_for_an_unknown_keyword(df):
    with pytest.raises(TypeError, match=r"^rsi\(\) got an unexpected keyword argument 'lenght'$"):
        df.copy().ta.strategy(ta.Strategy("typo", ta=[{"kind": "rsi", "lenght": 7}]))


def test_a_strategy_wide_keyword_reaches_only_the_indicators_that_accept_it(df):
    frame = df.copy()
    frame.ta.strategy("momentum", length=7)
    assert "RSI_7" in frame.columns and "RSI_14" not in frame.columns
    assert "MACD_12_26_9" in frame.columns  # macd has no length and is not passed one
    plain = df.copy()
    plain.ta.strategy(ta.Strategy("plain", ta=[{"kind": "rsi"}, {"kind": "macd"}]), length=7)
    assert {"RSI_7", "MACD_12_26_9"} <= set(plain.columns)


def test_a_strategy_wide_keyword_no_indicator_accepts_raises(df):
    with pytest.raises(TypeError, match=r"^strategy\(\) got keyword arguments that no indicator in 'momentum' accepts: 'lenght'$"):
        df.copy().ta.strategy("momentum", lenght=7)


def test_a_custom_indicator_gets_every_strategy_wide_keyword(df):
    seen = {}

    def kwtest(self, **kwargs):
        seen.update(kwargs)
        return self._df["close"].rename("KWTEST")

    with (
        mock.patch.object(ta.AnalysisIndicators, "kwtest", kwtest, create=True),
        mock.patch.dict(ta.Category, {"trend": [*ta.Category["trend"], "kwtest"]}),
    ):
        df.copy().ta.strategy(ta.Strategy("custom", ta=[{"kind": "kwtest"}]), anything=1)
    assert seen == {"anything": 1, "append": True}
