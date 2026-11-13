from functools import lru_cache
from pathlib import Path

from pandas import read_csv

SAMPLE_CSV = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"


@lru_cache(maxsize=1)
def _read_sample_csv():
    df = read_csv(
        SAMPLE_CSV,
        index_col="date",
        parse_dates=True,
    )
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def get_sample_data():
    """SPY_D, read once; a copy the caller may change. Every test and fixture generator loads it here."""
    return _read_sample_csv().copy()
