"""Shared harness of the registry-wide contract tests.

Each registry-wide contract runs every registered indicator against one class
of input: every ``test_*_contract.py`` module, ``test_degenerate_input.py``
and ``test_oracle_registry.py``, and the registry sweeps of the other test
modules. They share these pieces, kept here so they cannot drift apart:

* the data: :func:`spy` returns the last *bars* rows of ``SPY_D``;
* the registry: :data:`INDICATORS`, :data:`OHLCV_INDICATORS`, :func:`inputs`
  and :func:`call`, which feed an indicator the OHLCV columns its signature
  names, and :func:`candidate_calls`, its keyword variants;
* the groups several contracts exempt for the same reason (:data:`ELEMENTWISE`,
  :data:`CUMULATIVE`, :data:`WHOLE_SERIES`, :data:`LOOKAHEAD`);
* known exceptions: :func:`params` and :func:`pin` turn a table of
  ``{key: reason}`` into strict xfails, so an entry that starts passing fails
  and the table cannot go stale. They expect an ``AssertionError``: a known
  entry that crashes with anything else fails instead of counting as expected.

Every reason starts with one of :data:`REASON_PREFIXES`:

* ``OPEN:`` a defect, to be fixed;
* ``KEPT:`` a decided behaviour that breaks the contract on purpose;
* ``TALIB:`` a difference that is TA-Lib's, not ours.
"""

from __future__ import annotations

import ast
import inspect
import typing
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.config import get_sample_data

COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
INDICATORS = sorted({name for names in ta.Category.values() for name in names})
# Not a function of one OHLCV frame: two-series math, signal helpers, a
# required benchmark or periods series, the ma() dispatcher.
NOT_OHLCV = {"add", "sub", "mult", "div", "long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma"}
# vp returns price bins, not one row per bar.
NOT_TIME_SERIES = {"vp"}
OHLCV_INDICATORS = sorted(set(INDICATORS) - NOT_OHLCV - NOT_TIME_SERIES)

# Elementwise math: an indicator only in the sense of being registered.
ELEMENTWISE = {"acos", "asin", "atan", "ceil", "cos", "cosh", "exp", "floor", "ln", "log10", "npabs", "npround"}
ELEMENTWISE |= {"sin", "sinh", "sqrt", "tan", "tanh", "todeg", "torad", "trunc"}
# Running totals: a bar's contribution stays in every later value.
CUMULATIVE = {"ad", "aobv", "nvi", "obv", "pvi", "pvt", "wad"}
# A fit over the entire series: one changed bar moves every value.
WHOLE_SERIES = {"tos_stdevall"}
# Non-causal by design at their defaults (see test_lookahead.py): a bar depends on later bars.
LOOKAHEAD = {"dpo", "ichimoku", "tos_stdevall"}
# Columns no price-transform contract (scale, shift) compares, with the reason;
# the rest of their indicator still is. Not a dependence on scale or level:
# moving the prices rounds a comparison that sits exactly on its threshold to
# the other side.
BOUNDARY_COLUMNS = {"CDL_HIGHWAVE": "flips 1-2 bars that sit exactly on its threshold; TA-Lib's CDLHIGHWAVE flips the same bars"}
# Boolean parameters candidate_calls() leaves alone: talib selects a different
# implementation altogether, and whether TA-Lib is installed varies by environment.
SKIP_BOOL_KWARGS = {"talib"}

REASON_PREFIXES = ("OPEN:", "KEPT:", "TALIB:")


def spy(bars: int | None = None) -> pd.DataFrame:
    """The last *bars* rows of SPY_D (all of it for None), as a copy the caller may change."""
    df = get_sample_data()
    return df if bars is None else df.iloc[-bars:].copy()


def inputs(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """The keyword arguments that feed indicator *name* from *df*: the OHLCV columns its signature names,
    or, for an indicator in :data:`NOT_OHLCV`, inputs derived from them."""
    params = inspect.signature(getattr(ta, name)).parameters
    if name == "ma":
        return {"name": "ema", "source": df["close"]}
    if name == "mavp":
        return {"close": df["close"], "periods": pd.Series(10.0, index=df.index)}
    if name in ("beta", "correl"):
        return {"close": df["close"], "benchmark": df["open"]}
    if name in ("add", "sub", "mult", "div"):
        a, b = list(params)[:2]
        return {a: df["close"], b: df["open"]}
    if name in ("long_run", "short_run"):
        return {"fast": ta.ema(df["close"], 5), "slow": ta.ema(df["close"], 20)}
    if name == "tsignals":
        return {"trend": (df["close"] > ta.sma(df["close"], 20)).astype(int)}
    if name == "xsignals":
        return {"signal": ta.rsi(df["close"]), "xa": 70, "xb": 30}
    return {p: df[c] for p, c in COLUMNS.items() if p in params}


def call(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run indicator *name* on :func:`inputs` from *df*; a Series comes back as a frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = getattr(ta, name)(**inputs(name, df))
    return result.to_frame() if isinstance(result, pd.Series) else result


def values(name: str, df: pd.DataFrame) -> np.ndarray:
    """:func:`call` as a float array, missing values as NaN."""
    return call(name, df).to_numpy(float, na_value=np.nan)


def boolean_kwargs(name: str) -> list[str]:
    """Names of the boolean parameters of indicator *name*."""
    func = getattr(ta, name, None)
    if func is None:
        return []
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []

    found = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in SKIP_BOOL_KWARGS:
            continue
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            continue
        optional_of = typing.get_args(annotation)
        if annotation is bool or (bool in optional_of and type(None) in optional_of):
            found.append(parameter_name)
    return found


def candidate_calls(name: str) -> list[dict]:
    """The default call plus each boolean keyword of *name* flipped on its own."""
    return [{}] + [{keyword: value} for keyword in boolean_kwargs(name) for value in (True, False)]


_DEFAULT_HELPERS = {"_pos_int", "_pos_float", "_number", "_bool_param", "_str_param"}


def code_defaults(source: str) -> dict:
    """{parameter: default} read from the validation calls in an indicator's *source*
    (``length = _pos_int(length, 10, "length")`` gives ``{"length": 10}``)."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        if name in _DEFAULT_HELPERS and len(call.args) >= 2 and isinstance(call.args[0], ast.Name):
            try:
                found.setdefault(call.args[0].id, ast.literal_eval(call.args[1]))
            except ValueError:  # computed defaults (e.g. lensig defaults to length)
                pass
        elif name == "get_drift" and call.args and isinstance(call.args[0], ast.Name):
            found.setdefault(call.args[0].id, 1)
    return found


def _xfail(reason: str) -> pytest.MarkDecorator:
    if not reason.startswith(REASON_PREFIXES):
        raise ValueError(f"a known-exception reason must start with one of {REASON_PREFIXES}, got {reason!r}")
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def params(
    rows: Iterable[tuple],
    known: Mapping[Hashable, str],
    *,
    key: Callable[[tuple], Hashable] = lambda row: row,
    id: Callable[[tuple], str] = lambda row: "-".join(map(str, row)),
) -> list[Any]:
    """One ``pytest.param`` per row; a row whose key is in *known* is a strict xfail with its reason."""
    out = []
    for row in rows:
        reason = known.get(key(row))
        out.append(pytest.param(*row, marks=[_xfail(reason)] if reason else [], id=id(row)))
    return out


def pin(request: pytest.FixtureRequest, known: Mapping[Hashable, str], key: Hashable) -> None:
    """:func:`params` for a test whose parameters are built elsewhere: mark this run when *key* is known.

    A reason without a valid prefix raises here, when the test runs, not at collection as with :func:`params`.
    """
    reason = known.get(key)
    if reason:
        request.applymarker(_xfail(reason))
