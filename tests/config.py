from functools import lru_cache
from pathlib import Path

from pandas import read_csv


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
