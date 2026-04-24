import os
from functools import lru_cache
from pandas import read_csv

VERBOSE = True

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable


@lru_cache(maxsize=1)
def _read_sample_csv():
    df = read_csv(
        "data/SPY_D.csv",
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
