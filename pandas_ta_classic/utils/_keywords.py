"""The keywords each indicator accepts (issue #340).

An indicator's ``**kwargs`` used to take any name, so a misspelled option
(``rsi(close, lenght=7)``) returned the default result without a word. The
outermost indicator call now checks its keywords against this table and
raises ``TypeError`` for any other name.

An indicator accepts its signature, ``fillna``/``fill_method``, the options
below that it reads from ``**kwargs`` or hands on to an inner function, and,
when it computes with a moving average or a candle pattern chosen by a
parameter, that choice's own options. ``tests/test_keyword_table.py`` keeps
the tables in step with the code.

Only the package's own indicators are checked: an indicator loaded with
``custom.import_dir()`` or ``custom.bind()`` follows its own code.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import Any

_FILL = frozenset({"fillna", "fill_method"})
_EMA = frozenset({"adjust", "sma"})
_MIN_PERIODS = frozenset({"min_periods"})
_SIGNALS = frozenset({"signal_indicators", "xa", "xb", "xserie", "xserie_a", "xserie_b", "cross_values", "cross_series"})

# The options an indicator reads from **kwargs, or hands on to an inner function
# through them, besides its signature and fillna/fill_method. Keyed by function
# name (df.ta.max runs rolling_max). as_dataframe, interval and trend_reset were
# removed: they stay listed so the indicator raises its own TypeError naming the
# replacement.
_KWARGS_OPTIONS: dict[str, frozenset[str]] = {
    "aobv": frozenset({"run_length", "talib"}),
    "atr": frozenset({"percent"}),
    "beta": _MIN_PERIODS,
    "cdl_doji": frozenset({"naive"}),
    "cmf": _MIN_PERIODS,
    "correl": _MIN_PERIODS,
    "cpr": frozenset({"interval", "lookahead", "virgin_lookforward", "width_narrow", "width_wide"}),
    "dema": _EMA,
    "donchian": frozenset({"lower_min_periods", "upper_min_periods"}),
    "dpo": frozenset({"lookahead"}),
    "ema": _EMA,
    "emv": frozenset({"divisor"}),
    "er": _SIGNALS,
    "ichimoku": frozenset({"as_dataframe", "lookahead"}),
    "kc": frozenset({"tr"}),
    "kurtosis": _MIN_PERIODS,
    "linreg": frozenset({"angle", "degrees", "intercept", "r", "slope", "tsf"}),
    "macd": _SIGNALS | {"asmode"},
    "macdfix": _SIGNALS | {"asmode"},
    "mad": _MIN_PERIODS,
    "massi": _EMA | {"talib"},
    "median": _MIN_PERIODS,
    "midpoint": _MIN_PERIODS,
    "midprice": _MIN_PERIODS,
    "mmar": frozenset({"num_ribbons", "step"}),
    "msw": frozenset({"tulipy"}),
    "pvol": frozenset({"signed"}),
    "qstick": frozenset({"ma"}),
    "quantile": _MIN_PERIODS,
    "rainbow": frozenset({"num_ribbons"}),
    "rsi": _SIGNALS,
    "rsx": _SIGNALS,
    "skew": _MIN_PERIODS,
    "sma": _MIN_PERIODS,
    "squeeze": frozenset({"asint", "detailed", "lazybear", "tr"}),
    "squeeze_pro": frozenset({"asint", "detailed", "tr"}),
    "stc": frozenset({"ma1", "ma2", "osc"}),
    "stdev": _MIN_PERIODS,
    "t3": _EMA,
    "td_seq": frozenset({"show_all"}),
    "tema": _EMA,
    "tos_stdevall": frozenset({"lookahead"}),
    "trix": _EMA,
    "tsi": _EMA | {"talib"},
    "tsignals": frozenset({"trend_reset"}),
    "ui": frozenset({"everget"}),
    "variance": _MIN_PERIODS,
    "vortex": _MIN_PERIODS,
    "vp": frozenset({"lookahead", "sort_close"}),
    "willr": _MIN_PERIODS,
    "xsignals": frozenset({"trend_reset"}),
}

# The options each moving average of ma() reads besides length, offset and talib,
# for the indicators that hand them on to the MA they smooth with. Per MA, because
# a strategy-wide drift reaches every indicator and only vidya reads it.
_MA_OPTIONS: dict[str, frozenset[str]] = {
    "dema": _EMA,
    "ema": _EMA,
    "fwma": frozenset({"asc"}),
    "linreg": frozenset({"angle", "degrees", "intercept", "r", "slope", "tsf"}),
    "midpoint": _MIN_PERIODS,
    "sma": _MIN_PERIODS,
    "t3": _EMA | {"a"},
    "tema": _EMA,
    "vidya": frozenset({"drift"}),
    "wma": frozenset({"asc"}),
    "zlma": frozenset({"mamode"}),
}

# The moving averages of ma() that take talib.
_MA_TALIB = frozenset({"dema", "ema", "linreg", "midpoint", "sma", "t3", "tema", "trima", "wma"})

# Indicators that compute with the moving average one of their parameters names:
# (that parameter, its default, True when the indicator hands its **kwargs on to
# ma() whole, so the MA's talib reaches it too; kc, apo and ppo pass the MA's
# options only).
_MA_CHOICE: dict[str, tuple[str, str, bool]] = {
    "amat": ("mamode", "ema", True),
    "aobv": ("mamode", "ema", True),
    "apo": ("mamode", "sma", False),
    "bbands": ("mamode", "sma", True),
    "bias": ("mamode", "sma", True),
    "kc": ("mamode", "ema", False),
    "ppo": ("mamode", "sma", False),
    "qstick": ("ma", "sma", True),
}

# The options of the candle patterns cdl_pattern() hands its **kwargs on to.
_PATTERN_OPTIONS: dict[str, frozenset[str]] = {
    "doji": frozenset({"length", "factor", "asint", "naive"}),
    "inside": frozenset({"asbool"}),
    **dict.fromkeys(
        ("abandonedbaby", "darkcloudcover", "eveningdojistar", "eveningstar", "mathold", "morningdojistar", "morningstar"),
        frozenset({"penetration"}),
    ),
}


def _is_package_indicator(fn: Callable) -> bool:
    return fn.__module__.startswith("pandas_ta_classic.")


def static_keywords(fn: Callable) -> frozenset[str]:
    """The keywords a package indicator accepts whatever moving average or pattern it runs."""
    names = frozenset(inspect.signature(fn).parameters) - {"kwargs"}
    return names | _FILL | _KWARGS_OPTIONS.get(fn.__name__, frozenset())


def _ma_keywords(choice: Any, forwards_talib: bool) -> frozenset[str]:
    if not isinstance(choice, str):
        return frozenset()  # the indicator raises for it
    choice = choice.lower()
    talib = frozenset({"talib"}) if forwards_talib and choice in _MA_TALIB else frozenset()
    return _MA_OPTIONS.get(choice, frozenset()) | talib


def choice_keywords(name: str, arguments: Mapping[str, Any]) -> frozenset[str]:
    """The options of the moving average or the candle patterns *arguments* choose for indicator *name*."""
    if name == "ma":
        # ma() hands every keyword to the MA it dispatches to
        return frozenset({"length", "offset"}) | _ma_keywords(arguments.get("name") or "ema", True)
    if name in _MA_CHOICE:
        parameter, default, forwards_talib = _MA_CHOICE[name]
        return _ma_keywords(arguments.get(parameter) or default, forwards_talib)
    if name == "cdl_pattern":
        chosen = arguments.get("name")
        if chosen is None or chosen == "all":
            chosen = _PATTERN_OPTIONS
        elif isinstance(chosen, str):
            chosen = [chosen]
        return frozenset().union(*(_PATTERN_OPTIONS.get(pattern, frozenset()) for pattern in chosen))
    return frozenset()


def accepted_keywords(fn: Callable, kwargs: Mapping[str, Any]) -> frozenset[str] | None:
    """Every keyword *fn* accepts when called with *kwargs*.

    None for a function that is not one of the package's indicators and takes
    ``**kwargs`` (a custom indicator): its own code decides.
    """
    if not _is_package_indicator(fn):
        parameters = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return None
        return frozenset(parameters)
    return static_keywords(fn) | choice_keywords(fn.__name__, kwargs)


def check_keywords(fn: Callable, static: frozenset[str], sig: inspect.Signature, args: tuple, kwargs: Mapping[str, Any]) -> None:
    """Raise TypeError when the package indicator *fn* is called with a keyword it does not accept.

    *static* is ``static_keywords(fn)``, computed once by the caller.
    """
    unknown = set(kwargs) - static
    if not unknown:
        return
    # The choice may be passed by position: bbands(close, 20, 2.0, 0, "ema", adjust=True)
    bound = sig.bind_partial(*args, **kwargs).arguments
    arguments = {**bound.get("kwargs", {}), **bound}
    unknown -= choice_keywords(fn.__name__, arguments)
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        plural = "s" if len(unknown) > 1 else ""
        raise TypeError(f"{fn.__name__}() got {'an ' if not plural else ''}unexpected keyword argument{plural} {names}")
