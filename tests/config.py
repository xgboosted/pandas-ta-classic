import hashlib
import json
import os
import unittest
from pandas import read_csv

VERBOSE = True

try:
    import talib as tal

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    tal = None

talib_test = unittest.skipUnless(HAS_TALIB, "TA-Lib not installed")

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable


def get_sample_data():
    df = read_csv(
        "data/SPY_D.csv",
        index_col="date",
        parse_dates=True,
    )
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def error_analysis(df, kind, msg, icon=INFO, newline=True):
    if VERBOSE:
        s = f"{icon} {df.name}['{kind}']: {msg}"
        if newline:
            s = f"\n{s}"
        print(s)


def assert_offset(test_case, func, *args, expected_cols=None, none_arg_idx=0, **kwargs):
    """Assert result(offset=1) == result(offset=0).shift(1) for all columns.

    Also exercises fillna and fill_method code paths present in every indicator.

    Args:
        none_arg_idx: Index of the positional arg to replace with None when
            testing the None-guard branch. Defaults to 0 (first arg). Override
            when the indicator's primary verified series is not the first arg.
            Set to None to skip the None-guard assertion entirely (e.g. for
            indicators that return a tuple instead of None).
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "offset"}
    result_0 = func(*args, **kwargs_clean)
    result_1 = func(*args, offset=1, **kwargs_clean)

    # Exercise fillna / fill_method branches present in all indicators
    func(*args, **kwargs_clean, fillna=0)
    func(*args, **kwargs_clean, fill_method="ffill")
    func(*args, **kwargs_clean, fill_method="bfill")

    # Exercise None-guard branch (verify_series(None) → None → early return)
    if none_arg_idx is not None:
        none_args = list(args)
        none_args[none_arg_idx] = None
        test_case.assertIsNone(func(*none_args))

    import pandas as pd

    if isinstance(result_0, pd.Series):
        cols = [None]
    else:
        cols = expected_cols if expected_cols is not None else list(result_0.columns)

    for col in cols:
        s0 = result_0[col] if col is not None else result_0
        s1 = result_1[col] if col is not None else result_1
        pd.testing.assert_series_equal(s1, s0.shift(1), check_names=False)


def assert_nan_count(test_case, result, length):
    """Assert first (length-1) rows are NaN."""
    import pandas as pd

    if isinstance(result, pd.Series):
        head = result.iloc[: length - 1]
    else:
        head = result.iloc[: length - 1]
    test_case.assertTrue(
        head.isna().all().all() if hasattr(head, "columns") else head.isna().all()
    )


def assert_columns(test_case, result, expected_columns):
    """Assert DataFrame has exactly these columns."""
    test_case.assertListEqual(list(result.columns), list(expected_columns))


def hash_result(result):
    """Stable SHA-256 hash of a Series or DataFrame (8 decimal precision)."""
    import pandas as pd

    if isinstance(result, pd.Series):
        content = [
            round(float(x), 8) if not pd.isna(x) else None for x in result.values
        ]
    elif isinstance(result, pd.DataFrame):
        content = {
            col: [
                round(float(x), 8) if not pd.isna(x) else None
                for x in result[col].values
            ]
            for col in sorted(result.columns)
        }
    else:
        return None
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()
