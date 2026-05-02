from functools import lru_cache
from pathlib import Path
from pandas import read_csv

VERBOSE = False

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable


@lru_cache(maxsize=1)
def _read_sample_csv():
    csv_path = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"
    df = read_csv(
        csv_path,
        index_col="date",
        parse_dates=True,
    )
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def get_sample_data():
    return _read_sample_csv().copy()


def error_analysis(df, kind, msg, icon=INFO, newline=True):
    if VERBOSE:
        s = f"{icon} {df.name}['{kind}']: {msg}"
        if newline:
            s = f"\n{s}"
        print(s)


def assert_offset(test_case, func, args, **kwargs):
    """Verify offset=1 branch executes and returns a result."""
    test_case.assertIsNotNone(func(*args, offset=1, **kwargs))


def assert_fill(test_case, func, args, **kwargs):
    """Verify fillna and fill_method branches execute and return results."""
    test_case.assertIsNotNone(func(*args, fillna=0, **kwargs))
    test_case.assertIsNotNone(func(*args, fill_method="ffill", **kwargs))
    test_case.assertIsNotNone(func(*args, fill_method="bfill", **kwargs))


def assert_none_guard(test_case, func, args, none_arg_idx=0):
    """Verify function returns None when required arg is None."""
    none_args = list(args)
    none_args[none_arg_idx] = None
    test_case.assertIsNone(func(*none_args))
