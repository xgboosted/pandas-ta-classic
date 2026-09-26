"""No silent fallbacks (AGENTS.md "Fail-fast — No Fallbacks").

Each test pins a place that used to substitute a value, return None, log and
carry on, or hide an error, and now raises (or returns the honest result).
Every one of these used to "work" silently.
"""

import importlib
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.utils import df_year_to_date, is_datetime_ordered

logging.getLogger("pandas_ta_classic").setLevel(logging.CRITICAL)


@pytest.fixture
def spy():
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df.iloc[-300:].copy()


# --- A: a different computation or a made-up value ---------------------------


def test_zlma_unknown_mamode_raises(spy):
    with pytest.raises(ValueError, match=r"zlma\(\) mamode must be one of .*got 'smaa'"):
        ta.zlma(spy.close, mamode="smaa")  # used to compute an EMA


def test_jensens_alpha_raises_on_interior_gap_and_skips_the_leading_nan():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0003, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    returns = 0.0005 + 1.2 * bench
    bench.iloc[0] = returns.iloc[0] = np.nan  # pct_change's first bar
    assert ta.jensens_alpha(returns, bench) == pytest.approx(0.0005)
    gapped = bench.copy()
    gapped.iloc[150] = np.nan
    with pytest.raises(ValueError, match=r"benchmark_returns has 1 missing value"):
        ta.jensens_alpha(returns, gapped)  # used to interpolate it


def test_vwap_unordered_index_raises(spy):
    shuffled = spy.iloc[np.r_[0:100, 200:300, 100:200]]
    with pytest.raises(ValueError, match="ascending time order"):
        ta.vwap(shuffled.high, shuffled.low, shuffled.close, shuffled.volume)  # used to warn and compute


def test_is_datetime_ordered_checks_every_step(spy):
    assert is_datetime_ordered(spy)
    assert not is_datetime_ordered(spy.iloc[np.r_[0:100, 200:300, 100:200]])  # first < last, middle unsorted


def test_df_year_to_date_is_empty_without_rows_this_year(spy):
    assert df_year_to_date(spy).empty  # 2020 data; it used to return the whole frame


def test_version_is_not_made_up_when_the_package_is_missing():
    code = (
        "import sys, importlib.metadata as m\n"
        "sys.modules['pandas_ta_classic._version'] = None\n"
        "def missing(name): raise m.PackageNotFoundError(name)\n"
        "m.version = missing\n"
        "import pandas_ta_classic\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent.parent, check=False)
    assert result.returncode != 0 and "PackageNotFoundError" in result.stderr  # it used to report "0.0.0"


# --- B: a caller's mistake ignored ------------------------------------------


def test_missing_required_column_raises_and_missing_optional_one_is_skipped(spy):
    ohlc = spy[["open", "high", "low", "close"]]
    with pytest.raises(KeyError, match="'closee' not found"):
        ohlc.ta.sma(close="closee")  # used to log and return None
    with pytest.raises(KeyError, match="'volume' not found"):
        ohlc.ta.obv()
    assert "CPR_PIVOT" in ohlc.ta.cpr().columns  # cpr's volume is optional


def test_chained_custom_strategy_runs_in_order_with_several_workers(spy):
    """A chained entry reads a column an earlier entry appends.

    Pool workers each hold a copy of the input frame, so with chunks of one task
    (few cores) the chained ema never saw CUMLOGRET_1: it used to be dropped
    silently, and after the missing-column fix it raised KeyError. Ordered
    stages now run it after the entry that produces its input.
    """
    frame = spy.copy()
    frame.ta.cores = 2
    chain = ta.Strategy(
        "chain",
        [{"kind": "sma", "length": 10}, {"kind": "log_return", "cumulative": True}, {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"}],
    )
    frame.ta.strategy(chain)
    assert {"SMA_10", "CUMLOGRET_1", "EMA_5_CLR"} <= set(frame.columns)


def test_strategy_rejects_unknown_names_and_skips_unavailable_indicators(spy):
    frame = spy[["open", "high", "low", "close"]].copy()
    frame.ta.cores = 0
    with pytest.raises(ValueError, match=r"strategy\(\) got 'momentm', which is neither 'all' nor a category"):
        frame.ta.strategy("momentm")  # used to log and return None
    with pytest.raises(ValueError, match=r"exclude has unknown indicator name\(s\): \['smaa'\]"):
        frame.ta.strategy("overlap", exclude=["smaa"])
    with pytest.raises(TypeError, match="exclude must be a list"):
        frame.ta.indicators(as_list=True, exclude="sma")
    assert "sma" not in frame.ta.indicators(as_list=True, exclude=("sma",))  # a tuple used to be ignored
    frame.ta.strategy("volume")  # obv & co. need volume: skipped, not raised
    assert not any(c.startswith("OBV") for c in frame.columns)


def test_cdl_pattern_unknown_name_raises(spy):
    with pytest.raises(ValueError, match=r"unknown pattern\(s\): \['dojii'\]"):
        ta.cdl_pattern(spy.open, spy.high, spy.low, spy.close, name="dojii")  # used to return None


def test_col_names_count_must_match(spy):
    with pytest.raises(ValueError, match=r"col_names has 2 name\(s\) for 5 column\(s\)"):
        spy.ta.bbands(col_names=("A", "B"), append=True)  # used to log and append nothing
    with pytest.raises(ValueError, match="col_names has 2 names for one column"):
        spy.ta.sma(col_names=("A", "B"), append=True)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda d: d.ta.sma(append="no"), r"append must be True or False"),  # "no" used to append
        (lambda d: d.ta(kind="sma", timed="yes"), r"timed must be True or False"),
    ],
)
def test_flags_are_validated(spy, call, message):
    with pytest.raises(ValueError, match=message):
        call(spy)


# --- C: an error hidden -----------------------------------------------------


def test_candle_pattern_discovery_does_not_swallow_errors(monkeypatch):
    module = importlib.import_module("pandas_ta_classic.candles.cdl_pattern")  # the package attribute is the function
    real = importlib.import_module

    def broken(name, package=None):
        if name == ".cdl_2crows":
            raise RuntimeError("bug in a pattern module")
        return real(name, package)

    monkeypatch.setattr(module.importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="bug in a pattern module"):
        module._discover_native_patterns()  # used to log and drop the pattern


def test_missing_dependency_is_not_reported_as_a_missing_indicator(monkeypatch):
    import pandas_ta_classic._indicator_loader as loader

    def missing_dependency(name):
        raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")

    monkeypatch.setattr(loader, "_find_indicator_func", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        ta.__getattr__("zlma")  # used to become AttributeError: no attribute 'zlma'

    def missing_module(name):
        raise ModuleNotFoundError(f"No module named 'pandas_ta_classic.overlap.{name}'", name=f"pandas_ta_classic.overlap.{name}")

    monkeypatch.setattr(loader, "_find_indicator_func", missing_module)
    with pytest.raises(AttributeError):
        ta.__getattr__("zlma")

    def missing_internal_module(name):
        raise ModuleNotFoundError("No module named 'pandas_ta_classic.utils._gone'", name="pandas_ta_classic.utils._gone")

    monkeypatch.setattr(loader, "_find_indicator_func", missing_internal_module)
    with pytest.raises(ModuleNotFoundError, match="_gone"):
        ta.__getattr__("zlma")  # a broken import inside the module is not a missing indicator


def test_lazy_subpackage_does_not_hide_a_missing_dependency(monkeypatch):
    overlap = sys.modules["pandas_ta_classic.overlap"]

    def missing_dependency(name, package=None):
        raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        type(overlap).__getattr__(overlap, "pwma")  # used to become AttributeError


# --- D: broken or inconsistent ----------------------------------------------


def test_df_ta_lists_indicators_and_rejects_unknown_kinds(spy, capsys):
    spy.ta()  # used to raise AttributeError: no attribute 'help'
    assert "Indicators" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"kind='smaa' is not an indicator"):
        spy.ta(kind="smaa")
    with pytest.raises(TypeError, match="kind must be an indicator name"):
        spy.ta(kind=5)


def test_to_utc_property_converts_the_frame():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df.ta.to_utc  # noqa: B018  (a property with an effect, as documented)
    assert str(df.index.tz) == "UTC"  # it used to leave df unchanged
