import os
from pandas import DatetimeIndex, read_csv

VERBOSE = True

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable


def get_sample_data():
    df = read_csv(
        "data/SPY_D.csv",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        keep_date_col=True,
    )
    df.set_index(DatetimeIndex(df["date"]), inplace=True, drop=True)
    df.drop("date", axis=1, inplace=True)
    return df


def error_analysis(df, kind, msg, icon=INFO, newline=True):
    if VERBOSE:
        s = f"{icon} {df.name}['{kind}']: {msg}"
        if newline:
            s = f"\n{s}"
        print(s)
