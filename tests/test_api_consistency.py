"""Where the public API is not uniform, top-down: one table per question, one row per exception.

Every test below asks one question of every registered indicator -- which dtype
comes back, what an all-NaN input returns, whether passing a default explicitly
changes the column name, which names collide with something else -- and
compares the answer with a table at the top of this module. The table lists
the indicators that answer *differently from the majority*, with what they
answer instead. So reading the tables is reading the inconsistencies of the
API, and nothing can change silently:

- a new indicator that deviates fails the test until it is added to the table
  (or made to follow the convention);
- an indicator that is made consistent fails the test until its row is removed.

Every table is checked with a status, as the registry-wide contracts mark
their known exceptions:

- ``OPEN:`` a defect, or a question nobody has decided yet. A fix removes the
  row; the table shrinks as the API converges.
- ``KEPT:`` a decided exception the convention allows (an ``int8`` class
  column, ``vwap`` refusing an unsorted index, TA-Lib's +-200 hikkake).

A table can be ``OPEN:`` as a whole and keep single rows as ``KEPT:`` (and the
other way round), so a failure says whether a changed row was a defect or a
decision. The point is that the decision is made once, visibly, and not
rediscovered by the next user.

Three tables are measurements of hand-picked rows rather than registry sweeps,
so a new indicator never fails them: ``RATE_UNITS`` (which outputs are
measured), ``CANDLE_STANDALONE_DTYPE`` (which patterns), and the parameter
names ``PARAMETER_HOMONYMS`` asks about (every indicator taking one of them is
swept).

All checks run on the last 600 bars of ``SPY_D``; observed value sets (flag
encodings, ranges) are what that data produces.
"""

from __future__ import annotations

import builtins
import inspect
import math
import re
import sys
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.config import get_sample_data
from tests.test_docstring_defaults import _code_defaults

PACKAGE = Path(ta.__file__).parent
COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
OHLCV = {"open", "high", "low", "close", "volume"}
REGISTRY = {name: category for category, names in ta.Category.items() for name in names}
INDICATORS = sorted(REGISTRY)
# Elementwise math: an indicator only in the sense of being registered.
ELEMENTWISE = {"acos", "asin", "atan", "ceil", "cos", "cosh", "exp", "floor", "ln", "log10", "npabs", "npround"}
ELEMENTWISE |= {"sin", "sinh", "sqrt", "tan", "tanh", "todeg", "torad", "trunc"}
# Need arguments that are not OHLCV columns; the accessor test skips them.
SPECIAL_INPUT = {"ma", "mavp", "beta", "correl", "add", "sub", "mult", "div", "long_run", "short_run", "tsignals", "xsignals"}


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------

# The majority returns float64 in every column. These return integers, all of
# them flags or classes. BY DESIGN for the flags -- but note the split: the
# same kind of 0/1 or +-1 flag is float64 in psar, qqe, squeeze, supertrend,
# thermo and ttm_trend (see FLAG_COLUMNS).
NOT_FLOAT64 = {
    "amat": "int64",
    "aobv": "float64, int64",
    "cdl_doji": "int64",
    "cdl_inside": "int64",
    "cdl_pattern": "float64, int64",
    "cpr": "float64, int8",
    "decreasing": "int64",
    "ht_trendmode": "int64",
    "increasing": "int64",
    "long_run": "int64",
    "short_run": "int64",
    "smc_sweep": "int64",
    "tsignals": "int64",
    "xsignals": "int64",
}

# Every column with at most three distinct values on SPY, outside the candle
# patterns: dtype and value set. Four encodings for "a flag" coexist -- {0, 1},
# {-1, 1} without a neutral state, {-1, 0, 1}, and TA-Lib's {0, 100} -- in two
# dtypes, and with NaN warm-up (float) or 0 warm-up (int). OPEN as a question:
# which flags read 0 and which read NaN before they can know.
FLAG_COLUMNS = {
    "amat:AMATe_LR_8_21_2": "int64 {0, 1}",
    "amat:AMATe_SR_8_21_2": "int64 {0, 1}",
    "aobv:AOBV_LR_2": "int64 {0, 1}",
    "aobv:AOBV_SR_2": "int64 {0, 1}",
    "cdl_doji:CDL_DOJI_10_0.1": "int64 {0, 100}",
    "cdl_inside:CDL_INSIDE": "int64 {-1, 0, 1}",
    "cpr:CPR_POSITION": "int8 {-1, 0, 1}",
    "cpr:CPR_WIDTH_CLASS": "int8 {-1, 0, 1}",
    "decreasing:DEC_1": "int64 {0, 1}",
    "ht_trendmode:HT_TRENDMODE": "int64 {0, 1}",
    "increasing:INC_1": "int64 {0, 1}",
    "long_run:LR_2": "int64 {0, 1}",
    "psar:PSARr_0.02_0.2": "float64 {0, 1}",
    "qqe:QQEd_14_5_4.236": "float64 {-1, 1} +NaN",
    "short_run:SR_2": "int64 {0, 1}",
    "smc_sweep:SMC_SWEEP_15_1.5": "int64 {-1, 0, 1}",
    "squeeze:SQZ_NO": "float64 {0} +NaN",
    "squeeze:SQZ_OFF": "float64 {0, 1} +NaN",
    "squeeze:SQZ_ON": "float64 {0, 1} +NaN",
    "squeeze_pro:SQZPRO_NO": "float64 {0} +NaN",
    "squeeze_pro:SQZPRO_OFF": "float64 {0, 1} +NaN",
    "squeeze_pro:SQZPRO_ON_NARROW": "float64 {0} +NaN",
    "squeeze_pro:SQZPRO_ON_NORMAL": "float64 {0, 1} +NaN",
    "squeeze_pro:SQZPRO_ON_WIDE": "float64 {0, 1} +NaN",
    "supertrend:SUPERTd_7_3.0": "float64 {-1, 1} +NaN",
    "thermo:THERMOl_20_2_0.5": "float64 {0, 1} +NaN",
    "thermo:THERMOs_20_2_0.5": "float64 {0, 1} +NaN",
    "tsignals:TS_Entries": "int64 {0, 1}",
    "tsignals:TS_Exits": "int64 {0, 1}",
    "tsignals:TS_Trades": "int64 {-1, 0, 1}",
    "tsignals:TS_Trends": "int64 {0, 1}",
    "ttm_trend:TTM_TRND_6": "float64 {-1, 1} +NaN",
    "xsignals:TS_Entries": "int64 {0, 1}",
    "xsignals:TS_Exits": "int64 {0, 1}",
    "xsignals:TS_Trades": "int64 {-1, 0, 1}",
    "xsignals:TS_Trends": "int64 {0, 1}",
}

# Candle patterns follow TA-Lib: {-100, 0, 100}. OPEN: CDL_INSIDE reads +-1,
# and the same pattern has a different dtype standalone (cdl_doji, cdl_inside:
# int64) than inside cdl_pattern (float64, except CDL_INSIDE).
# CDL_HIKKAKE: BY DESIGN, TA-Lib marks a confirmed hikkake with +-200.
CANDLE_ENCODING_EXCEPTIONS = {"CDL_HIKKAKE": "float64 {-200, -100, 0, 100, 200}", "CDL_INSIDE": "int64 {-1, 0, 1}"}
CANDLE_STANDALONE_DTYPE = {"cdl_doji": ("int64", "float64"), "cdl_inside": ("int64", "int64")}

# float32 input: the majority upcasts to float64. OPEN: these keep float32 in
# every float column (a pandas expression end to end) ...
KEEPS_FLOAT32 = {
    "acos", "ad", "add", "asin", "atan", "avgprice", "bop", "ceil", "cos", "cosh", "cpr", "div", "drawdown", "emv", "exp", "floor", "hl2",
    "hlc3", "ln", "log10", "log_return", "marketfi", "medprice", "mom", "mult", "npabs", "npround", "nvi", "obv", "ohlc4", "pdist",
    "percent_return", "pvi", "pvol", "pvt", "roc", "rocp", "rocr", "rocr100", "sin", "sinh", "slope", "sqrt", "sub", "tan", "tanh", "todeg",
    "torad", "true_range", "trunc", "typprice", "vp", "vwap", "wcp",
}  # fmt: skip
# ... and these mix float32 and float64 columns in one result.
MIXED_FLOAT32 = {"aobv", "ha", "ichimoku", "thermo"}


# ---------------------------------------------------------------------------
# Data problems
# ---------------------------------------------------------------------------

# An input that is NaN on every bar: the majority returns NaN on every bar.
# These produce numbers from no data. Flags reading 0 are the open
# increasing/decreasing dtype decision; seeds (nvi/pvi 1000, psar's
# acceleration 0.02) and oscillators reading a fixed level (rsx) are OPEN.
INVENTS_VALUES_FROM_NAN = {
    "amat": "AMATe_LR_8_21_2=0, AMATe_SR_8_21_2=0",
    "aobv": "AOBV_LR_2=0, AOBV_SR_2=0",
    "cdl_pattern": "60 columns, values 0",
    "cpr": "CPR_POSITION=0, CPR_WIDTH_CLASS=0",
    "decay": "LDECAY_5=0",
    "decreasing": "DEC_1=0",
    "ebsw": "EBSW_40_10=0",
    "fisher": "FISHERT_9_1=0, FISHERTs_9_1=0",
    "ht_trendmode": "HT_TRENDMODE=0",
    "increasing": "INC_1=0",
    "long_run": "LR_2=0",
    "mfi": "MFI_14=0",
    "nvi": "NVI_1=1000",
    "psar": "PSARaf_0.02_0.2=0.02, PSARr_0.02_0.2=0",
    "pvi": "PVI_1=1000",
    "rsx": "RSX_14=0|50",
    "short_run": "SR_2=0",
    "smc_sweep": "SMC_SWEEP_15_1.5=0",
    "stc": "STC_10_12_26_0.5=0, STCstoch_10_12_26_0.5=0",
    "tsignals": "TS_Entries=0, TS_Exits=0, TS_Trades=0, TS_Trends=0",
    "vp": "neg_volume=0, pos_volume=0, total_volume=0",
    "wad": "WAD=0",
    "xsignals": "TS_Entries=0, TS_Exits=0, TS_Trades=0, TS_Trends=0",
}

# The set of output columns does not depend on the input for the majority.
# OPEN: cdl_pattern drops its doji column when the input is all NaN (and when
# it is shorter than the doji window).
COLUMNS_DEPEND_ON_INPUT = {"cdl_pattern": "all-NaN input drops CDL_DOJI_10_0.1"}

# A duplicated index label: the majority computes. OPEN: these raise pandas'
# "cannot reindex on an axis with duplicate labels" from an internal reindex.
DUPLICATE_INDEX_RAISES = {"dema", "kvo", "mfi", "stoch", "stochf", "t3", "tema"}
# An unsorted index: the majority computes on the bars in the order given.
# BY DESIGN for vwap, whose session anchor needs time order; the rest do not check.
UNSORTED_INDEX_RAISES = {"vwap"}


# ---------------------------------------------------------------------------
# Result metadata and naming
# ---------------------------------------------------------------------------

# result.category names the registry category for the majority. OPEN.
CATEGORY_ATTRIBUTE_MISMATCH = {"dm": "trend (registered under momentum)", "ttm_trend": "momentum (registered under trend)"}

# One output column comes back as a Series for the majority. OPEN.
SINGLE_COLUMN_DATAFRAME = {"ttm_trend"}

# Column names are UPPERCASE_<params>, optionally with a short lowercase
# qualifier (STOCHk, ATRr, PSARaf). These use words in mixed or lower case.
COLUMN_NAME_STYLE = {
    "aobv": "OBV_min_2, OBV_max_2",
    "cdl_z": "open_Z_30_1, high_Z_30_1, low_Z_30_1, close_Z_30_1",
    "ha": "HA_open, HA_high, HA_low, HA_close",
    "tsignals": "TS_Trends, TS_Trades, TS_Entries, TS_Exits",
    "vp": "low_close, mean_close, high_close, pos_volume, neg_volume, total_volume",
    "xsignals": "TS_Trends, TS_Trades, TS_Entries, TS_Exits",
}
FRAME_NAME_STYLE = {"ha": "Heikin-Ashi"}

# Passing a parameter's default explicitly leaves the result alone for 484 of
# 490 (indicator, parameter) pairs. OPEN: these format an int default as an int
# but an explicit value as a float, so kc() is KCLe_20_2 and kc(scalar=2) is
# KCLe_20_2.0 -- same values, two column names, two columns after append.
EXPLICIT_DEFAULT_RENAMES = {
    ("chop", "scalar"): "CHOP_14_1_100 -> CHOP_14_1_100.0",
    ("eom", "divisor"): "EOM_14_100000000 -> EOM_14_100000000.0",
    ("kc", "scalar"): "KCLe_20_2 -> KCLe_20_2.0",
    ("squeeze_pro", "kc_scalar_narrow"): "SQZPRO_20_2.0_20_2_1.5_1 -> SQZPRO_20_2.0_20_2_1.5_1.0",
    ("squeeze_pro", "kc_scalar_wide"): "SQZPRO_20_2.0_20_2_1.5_1 -> SQZPRO_20_2.0_20_2.0_1.5_1",
    ("thermo", "long"): "THERMO_20_2_0.5 -> THERMO_20_2.0_0.5",
}


# ---------------------------------------------------------------------------
# Signatures, docs and the accessor
# ---------------------------------------------------------------------------

# None selects the default for the majority. These put the value in the signature.
SIGNATURE_DEFAULTS = {
    "bbands": "ddof=0",
    "cdl_doji": "asint=True",
    "cdl_inside": "asbool=False",
    "cdl_pattern": "name='all'",
    "cpr": "method='classic', timeframe='daily', levels='standard', width_analysis=True, price_position=True, virgin_cpr=False",
    "dpo": "centered=True",
    "ichimoku": "include_chikou=True, append_span=False",
    "xsignals": "above=True, long=True",
}
NO_OFFSET = {"ma", "vp"}
NO_DOCSTRING = ELEMENTWISE | {
    "add",
    "sub",
    "mult",
    "div",
    "maxindex",
    "minindex",
    "minmax",
    "minmaxindex",
    "rolling_max",
    "rolling_min",
    "rolling_sum",
}
# df.ta.<name>() returns what the function returns when given every OHLCV
# column it accepts, names and values. OPEN: the accessor does not pass open_
# to ad, adosc, cmf and psl, so df.ta.ad() is AD while ta.ad(..., open_=df.open)
# is ADo, and the other three keep their names but compute without open_.
ACCESSOR_DIFFERS = {"ad": "AD vs ADo", "adosc": "values", "cmf": "values", "psl": "values"}


# ---------------------------------------------------------------------------
# Parameter names
# ---------------------------------------------------------------------------

# The window is `length` for the majority.
WINDOW_SYNONYMS = {"amat": "lookback", "msw": "period", "mavp": "periods"}
# Two spellings of "return integers/booleans instead of the default form".
OUTPUT_FORM_FLAGS = {
    "asint": {"cdl_doji", "decreasing", "increasing", "td_seq"},
    "asbool": {"cdl_inside", "tsignals", "xsignals"},
}
# One name, several meanings, read from the defaults: `scalar` is an output
# scale (100, 1) almost everywhere but a band width in kc; `k` is a window in
# stoch and a smoothing length in stochrsi; `factor` is a percentage, a
# multiplier and a smoothing weight; `c` a band width, CCI's constant and
# McGinley's constant.
PARAMETER_HOMONYMS = {
    "scalar": {"adx": 100, "adxr": 100, "aroon": 100, "bop": 1, "brar": 100, "cdl_doji": 100, "cdl_pattern": 100, "cfo": 100,
               "chop": 100, "cmo": 100, "dx": 100, "hwc": 1, "inertia": 100, "kc": 2, "natr": 100, "ppo": 100, "psl": 100,
               "pvo": 100, "roc": 100, "rsi": 100, "rvi": 100, "smi": 1, "trix": 100, "trixh": 100, "tsi": 100, "ui": 100},
    "k": {"stoch": 14, "stochrsi": 3},
    "factor": {"cdl_doji": 10, "qqe": 4.236, "stc": 0.5},
    "c": {"accbands": 4, "cci": 0.015, "mcgd": 1},
}  # fmt: skip

# Percent or fraction: measured against the plain-ratio definition, so a value
# of 100 means "percent" and 1 means "fraction". Both conventions are in use.
RATE_UNITS = {
    "roc": 100,
    "rocp": 1,
    "percent_return": 1,
    "bias": 1,
    "natr": 100,
    "hvol": 100,
    "avolume": 1,
    "BBB": 100,
    "BBP": 1,
    "DD_PCT": 1,
}


# ---------------------------------------------------------------------------
# Name collisions
# ---------------------------------------------------------------------------

# Two indicators writing the same column name. Aliases (same values) are
# harmless; OPEN: tsignals and xsignals write different values under the same
# names, so appending one overwrites the other without a word.
SHARED_COLUMNS = {
    "CDL_DOJI_10_0.1": ("cdl_doji, cdl_pattern", "same"),
    "CDL_INSIDE": ("cdl_inside, cdl_pattern", "same"),
    "DMN_14": ("adx, adxr", "same"),
    "DMP_14": ("adx, adxr", "same"),
    "EMA_10": ("ema, ma", "same"),
    "MAX_30": ("minmax, rolling_max", "same"),
    "MINUS_DM_14": ("dm, minus_dm", "same"),
    "MIN_30": ("minmax, rolling_min", "same"),
    "OBV": ("aobv, obv", "same"),
    "PLUS_DM_14": ("dm, plus_dm", "same"),
    "TS_Entries": ("tsignals, xsignals", "different"),
    "TS_Exits": ("tsignals, xsignals", "different"),
    "TS_Trades": ("tsignals, xsignals", "different"),
    "TS_Trends": ("tsignals, xsignals", "different"),
}
# Registered names that are also pandas Series/DataFrame methods with a
# different meaning (df.median() is one number, df.ta.median() a rolling median).
SHADOWS_PANDAS_METHOD = {"add", "div", "kurtosis", "median", "quantile", "skew", "squeeze", "sub"}
# Registered names that are also functions of the stdlib math module.
SHADOWS_STDLIB_MATH = {"acos", "asin", "atan", "ceil", "cos", "cosh", "exp", "floor", "log10", "sin", "sinh", "sqrt", "tan", "tanh", "trunc"}
# Names in the ta namespace that are Python builtins: the aliases of
# rolling_sum/max/min documented in docs/indicators.rst, so ta.max(close) is a
# 30-bar rolling maximum. They are not in __all__, so a star import does not
# shadow the builtins.
# (abs and round were registered as npabs and npround to avoid exactly this.)
SHADOWS_BUILTIN = {"max": "rolling_max", "min": "rolling_min", "sum": "rolling_sum"}
# Category subpackages named like a stdlib module or an OHLCV column.
CATEGORY_NAME_CLASHES = {"math": "stdlib module", "statistics": "stdlib module", "volume": "OHLCV column"}


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------


@cache
def _spy() -> pd.DataFrame:
    return get_sample_data().iloc[-600:]


def _kwargs(name: str, df: pd.DataFrame) -> dict[str, Any]:
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


def _frame(result: Any) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


def _call(name: str, df: pd.DataFrame | None = None, **extra: Any) -> Any:
    df = _spy() if df is None else df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(ta, name)(**_kwargs(name, df), **extra)


@cache
def _default(name: str) -> pd.DataFrame:
    return _frame(_call(name))


def _fmt(x: float) -> str:
    return f"{x:g}"


STATUSES = ("OPEN:", "KEPT:")


def _check(found: dict | set, pinned: dict | set, what: str, *, kept: frozenset = frozenset(), open_: frozenset = frozenset()) -> None:
    """``found`` must equal ``pinned``; report new rows, stale rows and changed rows separately.

    *what* starts with the table's status, ``OPEN:`` or ``KEPT:``; *kept* and
    *open_* name the rows whose status differs from the table's.
    """
    if not what.startswith(STATUSES):
        raise ValueError(f"a table's description must start with one of {STATUSES}, got {what!r}")
    unknown = (kept | open_) - set(pinned)
    if unknown:
        raise ValueError(f"{what}: status override for rows not in the table: {sorted(unknown, key=str)}")
    if isinstance(pinned, set):
        new, stale, changed = sorted(set(found) - pinned), sorted(pinned - set(found)), []
    else:
        new = sorted(set(found) - set(pinned), key=str)
        stale = sorted(set(pinned) - set(found), key=str)
        changed = sorted((k for k in set(found) & set(pinned) if found[k] != pinned[k]), key=str)
    lines = []
    if new:
        lines.append(f"new deviation, add it to the table or follow the convention: {[(k, found[k]) if isinstance(found, dict) else k for k in new]}")
    if stale:
        status = {k: "KEPT" if k in kept else "OPEN" if k in open_ else what.split(":")[0] for k in stale}
        lines.append(f"no longer deviates, remove the row: {[(k, status[k]) for k in stale]}")
    if changed:
        lines.append(f"row changed: {[(k, pinned[k], found[k]) for k in changed]}")
    assert not lines, f"{what}:\n  " + "\n  ".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_dtypes() -> None:
    found = {}
    for name in INDICATORS:
        dtypes = sorted({str(t) for t in _default(name).dtypes})
        if dtypes != ["float64"]:
            found[name] = ", ".join(dtypes)
    _check(found, NOT_FLOAT64, "KEPT: integer flags and classes; output dtype other than float64")


def test_flag_column_encodings() -> None:
    found = {}
    for name in INDICATORS:
        if name in ELEMENTWISE or name == "cdl_pattern":
            continue
        result = _default(name)
        for column in result.columns:
            values = result[column].dropna().to_numpy(float)
            uniq = np.unique(values)
            if 0 < uniq.size <= 3:
                nan = " +NaN" if result[column].isna().any() else ""
                found[f"{name}:{column}"] = f"{result[column].dtype} {{{', '.join(_fmt(v) for v in uniq)}}}{nan}"
    _check(found, FLAG_COLUMNS, "OPEN: which flags read 0 and which NaN before they can know; discrete columns (dtype and value set)")


def test_candle_pattern_encoding() -> None:
    patterns = _default("cdl_pattern")
    found = {}
    for column in patterns.columns:
        uniq = np.unique(patterns[column].dropna().to_numpy(float))
        if patterns[column].dtype != np.float64 or not set(uniq) <= {-100.0, 0.0, 100.0}:
            found[column] = f"{patterns[column].dtype} {{{', '.join(_fmt(v) for v in uniq)}}}"
    _check(found, CANDLE_ENCODING_EXCEPTIONS, "OPEN: candle pattern columns outside float64 {-100, 0, 100}", kept=frozenset({"CDL_HIKKAKE"}))
    standalone = {}
    for name in CANDLE_STANDALONE_DTYPE:
        alone = _default(name)
        standalone[name] = (str(alone.dtypes.iloc[0]), str(patterns[alone.columns[0]].dtype))
    _check(standalone, CANDLE_STANDALONE_DTYPE, "OPEN: one pattern, two dtypes; dtype standalone vs inside cdl_pattern (hand-picked rows)")


def test_float32_input() -> None:
    df32 = _spy().astype({c: "float32" for c in ["open", "high", "low", "close", "volume"]})
    keeps, mixed = set(), set()
    for name in INDICATORS:
        floats = {str(t) for t in _frame(_call(name, df32)).dtypes if str(t).startswith("float")}
        if floats == {"float32"}:
            keeps.add(name)
        elif floats == {"float32", "float64"}:
            mixed.add(name)
    _check(keeps, KEEPS_FLOAT32, "OPEN: one dtype rule for float32 input; float32 input stays float32 (majority: upcast to float64)")
    _check(mixed, MIXED_FLOAT32, "OPEN: float32 input gives mixed float32/float64 columns")


def test_all_nan_input() -> None:
    nan = _spy().copy()
    nan[:] = np.nan
    found = {}
    for name in INDICATORS:
        result = _frame(_call(name, nan))
        parts, seen = [], set()
        for column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce").dropna()
            if len(values):
                uniq = [_fmt(round(v, 6)) for v in np.unique(values.to_numpy(float))]
                seen.update(uniq)
                parts.append(f"{column}=" + "|".join(uniq))
        if parts:
            # cdl_pattern's 60 columns are summarised: how many, and which values.
            found[name] = f"{len(parts)} columns, values {'|'.join(sorted(seen))}" if name == "cdl_pattern" else ", ".join(sorted(parts))
    _check(found, INVENTS_VALUES_FROM_NAN, "OPEN: values produced from an all-NaN input")


def test_columns_do_not_depend_on_the_input() -> None:
    nan = _spy().copy()
    nan[:] = np.nan
    found = {}
    for name in INDICATORS:
        missing = [str(c) for c in _default(name).columns if c not in _frame(_call(name, nan)).columns]
        if missing:
            found[name] = "all-NaN input drops " + ", ".join(missing)
    _check(found, COLUMNS_DEPEND_ON_INPUT, "OPEN: output columns that depend on the input values")


@pytest.mark.parametrize(
    ("label", "pinned", "status", "message"),
    [
        ("duplicate", DUPLICATE_INDEX_RAISES, "OPEN:", "cannot reindex on an axis with duplicate labels"),
        ("unsorted", UNSORTED_INDEX_RAISES, "KEPT: the session anchor needs time order;", "must be in ascending time order"),
    ],
)
def test_index_order_and_uniqueness(label: str, pinned: set, status: str, message: str) -> None:
    df = _spy()
    if label == "duplicate":
        df = df.copy()
        df.index = df.index[:-1].append(df.index[-2:-1])
    else:
        df = df.sample(frac=1.0, random_state=0)
    raises = set()
    for name in INDICATORS:
        try:
            _call(name, df)
        except (ValueError, TypeError) as exc:
            # Any other error is a different defect: let it fail the test.
            if message not in str(exc):
                raise
            raises.add(name)
    _check(raises, pinned, f"{status} raises on a {label} index (majority: computes)")


def test_result_category_attribute() -> None:
    found = {}
    for name in INDICATORS:
        attr = getattr(_call(name), "category", None)
        if attr != REGISTRY[name]:
            found[name] = f"{attr} (registered under {REGISTRY[name]})"
    _check(found, CATEGORY_ATTRIBUTE_MISMATCH, "OPEN: result.category differs from the registry")


def test_single_output_is_a_series() -> None:
    found = {name for name in INDICATORS if isinstance(_call(name), pd.DataFrame) and _default(name).shape[1] == 1}
    _check(found, SINGLE_COLUMN_DATAFRAME, "OPEN: one column returned as a DataFrame")


def test_column_and_frame_name_style() -> None:
    word = re.compile(r"^[a-z]|_(?:[a-z]{2,}|[A-Z][a-z]{2,})(?:_|$)")
    columns = {}
    frames = {}
    for name in INDICATORS:
        bad = [c for c in _default(name).columns if word.search(str(c))]
        if bad:
            columns[name] = ", ".join(bad)
        result = _call(name)
        if isinstance(result, pd.DataFrame) and not re.fullmatch(r"[A-Z0-9_.]+[a-z]?[A-Z0-9_.]*", str(getattr(result, "name", ""))):
            frames[name] = str(result.name)
    _check(columns, COLUMN_NAME_STYLE, "OPEN: column names with lower- or mixed-case words")
    _check(frames, FRAME_NAME_STYLE, "OPEN: DataFrame .name outside the column-name alphabet")


def test_explicit_default_keeps_the_column_names() -> None:
    found = {}
    for name in INDICATORS:
        source = PACKAGE / REGISTRY[name] / f"{name}.py"
        params = inspect.signature(getattr(ta, name)).parameters
        base = _default(name)
        for param, value in _code_defaults(source.read_text(encoding="utf-8")).items():
            if param not in params or param == "offset":
                continue
            explicit = _frame(_call(name, **{param: value}))
            assert np.allclose(
                explicit.to_numpy(float, na_value=np.nan), base.to_numpy(float, na_value=np.nan), equal_nan=True
            ), f"{name}({param}={value!r}) changes the values, not just the name"
            if list(explicit.columns) != list(base.columns):
                found[(name, param)] = f"{base.columns[0]} -> {explicit.columns[0]}"
    _check(found, EXPLICIT_DEFAULT_RENAMES, "OPEN: passing the default explicitly renames the output")


def test_signature_shape() -> None:
    defaults = {}
    no_offset = set()
    for name in INDICATORS:
        params = inspect.signature(getattr(ta, name)).parameters
        explicit = [f"{p}={v.default!r}" for p, v in params.items() if v.default is not inspect.Parameter.empty and v.default is not None]
        if explicit:
            defaults[name] = ", ".join(explicit)
        if "offset" not in params:
            no_offset.add(name)
    _check(defaults, SIGNATURE_DEFAULTS, "OPEN: signature defaults other than None")
    _check(no_offset, NO_OFFSET, "OPEN: no offset parameter")


def test_docstring_present() -> None:
    found = {name for name in INDICATORS if "Args:" not in (getattr(ta, name).__doc__ or "")}
    _check(found, NO_DOCSTRING, "OPEN: no docstring with an Args: section")


def test_accessor_matches_the_function() -> None:
    df = _spy().copy()
    found = {}
    for name in INDICATORS:
        if name in SPECIAL_INPUT:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            via_accessor = _frame(getattr(df.ta, name)())
        function = _default(name)
        if list(via_accessor.columns) != list(function.columns):
            found[name] = f"{via_accessor.columns[0]} vs {function.columns[0]}"
        elif not np.allclose(via_accessor.to_numpy(float, na_value=np.nan), function.to_numpy(float, na_value=np.nan), equal_nan=True):
            found[name] = "values"
    _check(found, ACCESSOR_DIFFERS, "OPEN: df.ta.<name>() differs from the function given every OHLCV column")


def test_parameter_names() -> None:
    synonyms = {}
    for name in INDICATORS:
        params = inspect.signature(getattr(ta, name)).parameters
        for alias in ("period", "periods", "lookback", "window", "n"):
            if alias in params:
                synonyms[name] = alias
    _check(synonyms, WINDOW_SYNONYMS, "OPEN: window parameter not called length")
    for flag, pinned in OUTPUT_FORM_FLAGS.items():
        _check(
            {n for n in INDICATORS if flag in inspect.signature(getattr(ta, n)).parameters},
            pinned,
            f"OPEN: two spellings of one option; indicators taking {flag}",
        )
    for param, pinned_defaults in PARAMETER_HOMONYMS.items():
        found = {}
        for name in INDICATORS:
            if param in inspect.signature(getattr(ta, name)).parameters:
                found[name] = _code_defaults((PACKAGE / REGISTRY[name] / f"{name}.py").read_text(encoding="utf-8")).get(param)
        _check(found, pinned_defaults, f"OPEN: one name, several meanings; defaults of the parameter `{param}`")


def test_rate_units() -> None:
    df = _spy()
    close = df["close"]
    bands = _default("bbands")
    lower, mid, upper = (bands[c] for c in bands.columns[:3])
    log_ret = np.log(close / close.shift(1))
    measured = {
        "roc": (_call("roc"), close.pct_change(10)),
        "rocp": (_call("rocp"), close.pct_change(10)),
        "percent_return": (_call("percent_return"), close.pct_change(1)),
        "bias": (_call("bias"), close / ta.sma(close, 26) - 1),
        "natr": (_call("natr"), ta.atr(df["high"], df["low"], close) / close),
        "hvol": (_call("hvol"), log_ret.rolling(20).std(ddof=1) * np.sqrt(252)),
        "avolume": (_call("avolume"), log_ret.rolling(20).std(ddof=0) * np.sqrt(252)),
        "BBB": (bands[bands.columns[3]], (upper - lower) / mid),
        "BBP": (bands[bands.columns[4]], (close - lower) / (upper - lower)),
        "DD_PCT": (_default("drawdown")["DD_PCT"], 1 - close / close.cummax()),
    }
    units = {}
    for key, (value, plain) in measured.items():
        ratio = (value / plain).replace([np.inf, -np.inf], np.nan).dropna()
        units[key] = round(float(ratio.median()))
        assert np.allclose(ratio, units[key], rtol=1e-6), f"{key}: not a constant multiple of its plain ratio"
    _check(units, RATE_UNITS, "OPEN: percent (100) vs fraction (1)")


def test_output_column_collisions() -> None:
    owners: dict[str, list[str]] = {}
    for name in INDICATORS:
        for column in _default(name).columns:
            owners.setdefault(str(column), []).append(name)
    found = {}
    for column, names in owners.items():
        if len(names) > 1:
            values = [_default(n)[column].to_numpy(float, na_value=np.nan) for n in names]
            same = all(np.allclose(values[0], v, equal_nan=True) for v in values[1:])
            found[column] = (", ".join(names), "same" if same else "different")
    _check(
        found,
        SHARED_COLUMNS,
        "KEPT: aliases writing the same values; column names written by more than one indicator",
        open_=frozenset(k for k, (_, values) in SHARED_COLUMNS.items() if values == "different"),
    )
    lower: dict[str, set[str]] = {}
    for column in owners:
        lower.setdefault(column.lower(), set()).add(column)
    case_collisions = {k: sorted(v) for k, v in lower.items() if len(v) > 1}
    assert not case_collisions, f"column names that differ only in case: {case_collisions}"


def test_name_shadowing() -> None:
    _check(set(INDICATORS) & (set(dir(pd.Series)) | set(dir(pd.DataFrame))), SHADOWS_PANDAS_METHOD, "OPEN: registered names that are pandas methods")
    _check(set(INDICATORS) & set(dir(math)), SHADOWS_STDLIB_MATH, "OPEN: registered names that are stdlib math functions")
    builtin_names = {n: getattr(ta, n).__name__ for n in dir(ta) if not n.startswith("_") and n in dir(builtins)}
    _check(builtin_names, SHADOWS_BUILTIN, "KEPT: documented aliases outside __all__; ta.<builtin name> (value: what it really is)")
    clashes = {}
    for category in ta.Category:
        if category in sys.stdlib_module_names:
            clashes[category] = "stdlib module"
        elif category in OHLCV:
            clashes[category] = "OHLCV column"
    _check(clashes, CATEGORY_NAME_CLASHES, "OPEN: category names that clash")


def test_tables_name_registered_indicators() -> None:
    tables = [NOT_FLOAT64, KEEPS_FLOAT32, MIXED_FLOAT32, INVENTS_VALUES_FROM_NAN, COLUMNS_DEPEND_ON_INPUT, DUPLICATE_INDEX_RAISES, UNSORTED_INDEX_RAISES,
              CATEGORY_ATTRIBUTE_MISMATCH, SINGLE_COLUMN_DATAFRAME, COLUMN_NAME_STYLE, FRAME_NAME_STYLE, SIGNATURE_DEFAULTS, NO_OFFSET,
              NO_DOCSTRING, ACCESSOR_DIFFERS, WINDOW_SYNONYMS, CANDLE_STANDALONE_DTYPE]  # fmt: skip
    named = {k for t in tables for k in t} | {n for n, _ in EXPLICIT_DEFAULT_RENAMES} | {k.split(":")[0] for k in FLAG_COLUMNS}
    assert not named - set(REGISTRY), f"not registered: {sorted(named - set(REGISTRY))}"
