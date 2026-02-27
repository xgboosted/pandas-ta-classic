import os
from pandas import read_csv

VERBOSE = True

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


def assert_offset(test_case, func, *args, expected_cols=None, **kwargs):
    """Assert result(offset=1) == result(offset=0).shift(1) for all columns."""
    result_0 = func(*args, **kwargs)
    result_1 = func(*args, offset=1, **kwargs)

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
    test_case.assertTrue(head.isna().all().all() if hasattr(head, "columns") else head.isna().all())


def assert_columns(test_case, result, expected_columns):
    """Assert DataFrame has exactly these columns."""
    test_case.assertListEqual(list(result.columns), list(expected_columns))
