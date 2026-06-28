"""Lazy indicator function loader and AnalysisIndicators wrapper factory."""

import functools
import importlib
import inspect
from typing import Any, Callable, Optional

from pandas_ta_classic._meta import Category, _MATH_ALIASES

# Maps function param name → DataFrame column key
_COLUMN_PARAM_TO_COL_KEY: dict[str, str] = {
    "close": "close",
    "high": "high",
    "low": "low",
    "open_": "open",  # param 'open_' reads column 'open'
    "volume": "volume",
}

# Extended column params with multi-alias pop logic: {param_name: (aliases_to_pop, default_col)}
_SERIES_COLUMN_PARAMS: dict[str, tuple] = {
    "series_a": (("a", "series_a"), "close"),
    "series_b": (("b", "series_b"), "close"),
}

# Reverse lookup: indicator name → category name (built once at import)
_INDICATOR_TO_CATEGORY: dict[str, str] = {ind: cat for cat, indicators in Category.items() for ind in indicators}
_INDICATOR_TO_CATEGORY.update({alias: "math" for alias in _MATH_ALIASES})


def _find_indicator_func(name: str) -> Optional[Callable]:
    """Lazy-import and return the indicator function, or None if unknown."""
    cat = _INDICATOR_TO_CATEGORY.get(name)
    if cat is None:
        return None
    canonical = _MATH_ALIASES.get(name, name)
    mod = importlib.import_module(f"pandas_ta_classic.{cat}.{canonical}")
    func = getattr(mod, canonical, None)
    if func is None:
        raise AttributeError(f"module 'pandas_ta_classic.{cat}.{canonical}' has no attribute '{canonical}'")
    return func


def _make_ta_wrapper(func: Callable) -> Callable:
    """
    Build a bound-method wrapper for *func* suitable for AnalysisIndicators.

    Column params (close, high, low, open_, volume) are extracted from the
    host DataFrame via self._get_column; all other params flow through as
    positional *args (mapped to their non-column parameter names) or **kwargs.
    """
    sig = inspect.signature(func)
    # A column param is "required" (always fetched from the DataFrame) when:
    #   - it has no default value in the function signature, OR
    #   - it is one of the primary OHLCV columns (close, high, low, volume).
    # open_ is optional: only fetched when (a) it has no default (required by
    # the function) or (b) the caller explicitly passes open=<col> in kwargs.
    _ALWAYS_FETCH = {"close", "high", "low", "volume"}
    col_params_required = [
        p for p in sig.parameters if p in _COLUMN_PARAM_TO_COL_KEY and (p in _ALWAYS_FETCH or sig.parameters[p].default is inspect.Parameter.empty)
    ]
    col_params_optional = [
        p
        for p in sig.parameters
        if p in _COLUMN_PARAM_TO_COL_KEY and p not in _ALWAYS_FETCH and sig.parameters[p].default is not inspect.Parameter.empty
    ]
    # Ordered list of non-column positional/keyword parameters (for *args binding)
    non_col_positional = [
        pname
        for pname, param in sig.parameters.items()
        if pname not in _COLUMN_PARAM_TO_COL_KEY
        and pname not in _SERIES_COLUMN_PARAMS
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]

    # Series column params present in this function's signature
    series_col_params = [p for p in sig.parameters if p in _SERIES_COLUMN_PARAMS]

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_kwargs: dict[str, Any] = {}
        # Map positional args to their non-column parameter names
        for i, arg in enumerate(args):
            if i < len(non_col_positional):
                call_kwargs[non_col_positional[i]] = arg
        # Always extract required column values from DataFrame
        for param_name in col_params_required:
            col_key = _COLUMN_PARAM_TO_COL_KEY[param_name]
            col_val = kwargs.pop(col_key, col_key)
            call_kwargs[param_name] = self._get_column(col_val)
        # Extract optional column values only when explicitly requested by user
        for param_name in col_params_optional:
            col_key = _COLUMN_PARAM_TO_COL_KEY[param_name]
            if col_key in kwargs:
                col_val = kwargs.pop(col_key)
                call_kwargs[param_name] = self._get_column(col_val)
        # Handle series_a / series_b: pop aliases in order, fallback to default column
        for param_name in series_col_params:
            aliases, default_col = _SERIES_COLUMN_PARAMS[param_name]
            col_val = default_col
            for alias in aliases:
                if alias in kwargs:
                    col_val = kwargs.pop(alias)
                    break
            # Clean up any remaining aliases that weren't used
            for alias in aliases:
                kwargs.pop(alias, None)
            call_kwargs[param_name] = self._get_column(col_val)
        # Fill in None for any required non-column positional args not yet provided
        # (allows functions like long_run/short_run to return None when called with
        # no fast/slow args, which _post_process then converts to self._df)
        for pname, param in sig.parameters.items():
            if (
                pname not in _COLUMN_PARAM_TO_COL_KEY
                and pname not in _SERIES_COLUMN_PARAMS
                and pname not in call_kwargs
                and pname not in kwargs
                and param.default is inspect.Parameter.empty
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                )
            ):
                call_kwargs[pname] = None
        result = func(**call_kwargs, **kwargs)
        return self._post_process(result, **kwargs)

    # Rewrite __signature__: drop col params (they come from the DataFrame),
    # prepend 'self', keep everything else (length, offset, **kwargs, etc.)
    new_params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for pname, param in sig.parameters.items():
        if pname not in _COLUMN_PARAM_TO_COL_KEY and pname not in _SERIES_COLUMN_PARAMS:
            new_params.append(param)
    wrapper.__signature__ = sig.replace(parameters=new_params)
    return wrapper
