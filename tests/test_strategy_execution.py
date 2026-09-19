"""How strategy() executes: stages, serial/parallel parity, and argument checks.

The chunked Multiprocessing Pool that these replace decided correctness by
batching: a custom entry reading an earlier entry's column only saw it when
both happened to land in the same chunk, and produced no column otherwise,
without an error. The parity tests below run the same strategies serially and
on an Executor and demand identical frames.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

# A producer and its consumer. The consumer reads SMA_10, so it cannot run in
# the same stage as the indicator that creates it.
CHAINED = [
    {"kind": "sma", "length": 10},
    {"kind": "ema", "close": "SMA_10", "length": 5, "suffix": "CLR"},
]
# The same chain behind an unrelated indicator. This shifted the chunk
# boundary, which used to lose both chained columns at cores=4 and cores=8.
CHAINED_OFFSET = [{"kind": "wma", "length": 30}, *CHAINED]


def sample_frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, rows))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.2, rows),
            "high": close + rng.uniform(0.1, 1.0, rows),
            "low": close - rng.uniform(0.1, 1.0, rows),
            "close": close,
            "volume": rng.integers(1_000, 100_000, rows).astype(float),
        },
        index=pd.date_range("2020-01-01", periods=rows, freq="D"),
    )


def _strategy_in_worker(_ignored) -> int:
    """Run a strategy from inside a daemonic Pool worker.

    Nesting used to raise 'daemonic processes are not allowed to have
    children'; strategy() now runs serially there.
    """
    df = sample_frame(200)
    df.ta.cores = 4
    df.ta.strategy("momentum", cores=4)
    return len(df.columns)


@pytest.fixture(scope="module", autouse=True)
def lean_workers():
    """Spawned children inherit this, as docs/strategies.rst recommends.

    Every fresh interpreter commits about 750 MB for OpenBLAS thread buffers
    that the indicators never use. This module is the only one that starts
    processes, and under "pytest -n auto" each xdist worker starts its own:
    without this, peak system commit went from 85% to 97% on a 32-core machine.
    """
    previous = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    yield
    if previous is None:
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = previous


@pytest.fixture(scope="module")
def executor():
    # One pool for the whole module; per test it would be seven.
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
        yield pool


def assert_same_frame(serial: pd.DataFrame, parallel: pd.DataFrame) -> None:
    assert list(serial.columns) == list(parallel.columns)
    for column in serial.columns:
        pd.testing.assert_series_equal(serial[column], parallel[column])


@pytest.mark.parametrize(
    "strategy",
    [
        "all",
        "momentum",
        pytest.param(ta.Strategy("Chain", CHAINED), id="chained-custom"),
        pytest.param(ta.Strategy("ChainOffset", CHAINED_OFFSET), id="chained-custom-offset"),
    ],
)
def test_executor_matches_serial(strategy, executor):
    serial = sample_frame()
    serial.ta.strategy(strategy, cores=0)

    parallel = sample_frame()
    parallel.ta.strategy(strategy, executor=executor)

    assert_same_frame(serial, parallel)


@pytest.mark.parametrize("entries", [CHAINED, CHAINED_OFFSET])
def test_chained_columns_survive_parallel_execution(entries, executor):
    df = sample_frame()
    df.ta.strategy(ta.Strategy("Chain", entries), executor=executor)
    assert "SMA_10" in df.columns
    assert "EMA_5_CLR" in df.columns


def test_chained_column_is_not_satisfied_by_a_prefix_match(executor):
    """A stage boundary must not be decided by _get_column()'s fallback match.

    _get_column() falls back to a case-insensitive prefix match for a misspelled
    column, so an unrelated column can satisfy the name a chained entry reads --
    'MA' resolves to 'MACD_12_26_9'. Counting that as present would run the
    consumer in its producer's stage, where the worker frame does not carry the
    exact name yet, and it would silently read the other column instead.
    """
    entries = [{"kind": "sma", "length": 10}, {"kind": "ema", "close": "SMA_10", "length": 5, "suffix": "CLR"}]

    serial = sample_frame()
    serial["SMA_10_OLD"] = -999.0  # prefix-matches SMA_10, and is not it
    serial.ta.strategy(ta.Strategy("Chain", entries), cores=0)

    parallel = sample_frame()
    parallel["SMA_10_OLD"] = -999.0
    parallel.ta.strategy(ta.Strategy("Chain", entries), executor=executor)

    assert_same_frame(serial, parallel)
    assert (parallel["EMA_5_CLR"].dropna() != -999.0).all()


def test_concurrent_strategies_in_one_process_do_not_trip_the_guard():
    """The re-entrancy guard lives in os.environ, which is process-global.

    Testing the variable rather than the pid it holds made a second thread
    calling strategy(cores=...) fail with the __main__ guard message, which is
    not its problem. A spawned child inherits the parent's pid, so comparing
    still catches the recursion it is there for.
    """
    import threading

    failures: list[BaseException] = []

    def run() -> None:
        try:
            frame = sample_frame(200)
            frame.ta.strategy("momentum", cores=2)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, [repr(exc) for exc in failures]
    assert os.environ.get("_PANDAS_TA_CLASSIC_STRATEGY_PID") is None


def test_cores_kwarg_matches_serial():
    """strategy(cores=N) was accepted and dropped; only df.ta.cores was read."""
    serial = sample_frame()
    serial.ta.strategy(ta.Strategy("Chain", CHAINED_OFFSET), cores=0)

    parallel = sample_frame()
    parallel.ta.strategy(ta.Strategy("Chain", CHAINED_OFFSET), cores=2)

    assert_same_frame(serial, parallel)


def test_cores_defaults_to_serial():
    assert sample_frame().ta.cores == 0


def test_accessor_settings_reach_the_workers(executor):
    """df.ta.adjusted redirects 'close'; the worker frame must carry it.

    It travels in DataFrame.attrs, which pandas keeps through the column
    selection that slims the worker payload -- so nothing copies it explicitly.
    """
    def with_adjusted_column(frame):
        # Not a scaled close: RSI is scale invariant, so close * 1.5 would only
        # move the last bits and the check below would prove nothing.
        frame["adj_close"] = frame["close"].to_numpy()[::-1]
        return frame

    serial = with_adjusted_column(sample_frame())
    serial.ta.adjusted = "adj_close"
    serial.ta.strategy("momentum", cores=0)

    parallel = with_adjusted_column(sample_frame())
    parallel.ta.adjusted = "adj_close"
    parallel.ta.strategy("momentum", executor=executor)

    assert_same_frame(serial, parallel)

    # ... and the setting reached the workers because it changed the result.
    plain = with_adjusted_column(sample_frame())
    plain.ta.strategy("momentum", cores=0)
    assert not plain["RSI_14"].equals(serial["RSI_14"])


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_cores_kwarg_rejects_invalid_values(bad):
    df = sample_frame(200)
    with pytest.raises(ValueError, match=r"strategy\(\) cores must be an integer >= 0"):
        df.ta.strategy("momentum", cores=bad)


def test_executor_kwarg_rejects_non_executor():
    df = sample_frame(200)
    with pytest.raises(TypeError, match=r"strategy\(\) executor must be a concurrent.futures.Executor or None"):
        df.ta.strategy("momentum", executor="nope")


def test_unknown_strategy_name_raises():
    """df.ta.strategy("CommonStrategy") logged an invisible error and added nothing."""
    df = sample_frame(200)
    with pytest.raises(ValueError, match=r"strategy\(\) got 'CommonStrategy', which is neither 'all' nor a category"):
        df.ta.strategy("CommonStrategy")


def test_strategy_entry_shape_is_checked_where_it_is_written():
    """These used to surface only on the run, as KeyError or 'not a mapping'."""
    with pytest.raises(ValueError, match=r"Strategy 'S' entry 1 has no 'kind'"):
        ta.Strategy("S", [{"kind": "rsi"}, {"length": 10}])
    with pytest.raises(TypeError, match=r"Strategy 'S' entry 0: expected a dict"):
        ta.Strategy("S", ["sma"])
    with pytest.raises(TypeError, match=r"Strategy 'S' entry 0: 'kind' must be an indicator name"):
        ta.Strategy("S", [{"kind": 42}])


def test_unknown_indicator_in_a_custom_strategy_names_the_entry():
    df = sample_frame(200)
    strategy = ta.Strategy("S", [{"kind": "rsi"}, {"kind": "not_an_indicator"}])
    with pytest.raises(ValueError, match=r"Strategy 'S' entry 1: 'not_an_indicator' is not an indicator"):
        df.ta.strategy(strategy, cores=0)


def test_too_few_col_names_raises():
    """bbands with one col_name added no column at all, and said nothing."""
    df = sample_frame(200)
    strategy = ta.Strategy("S", [{"kind": "bbands", "length": 20, "col_names": ("BBL",)}])
    with pytest.raises(ValueError, match=r"col_names has 1 name\(s\) for \d+ columns"):
        df.ta.strategy(strategy, cores=0)


def test_exclude_must_be_a_list_of_names():
    """exclude='rsi' extended the exclusion list by 'r', 's', 'i' -- that is, nothing."""
    df = sample_frame(200)
    with pytest.raises(TypeError, match=r"strategy\(\) exclude must be a list of indicator names, got str"):
        df.ta.strategy("momentum", exclude="rsi", cores=0)
    with pytest.raises(TypeError, match=r"strategy\(\) exclude must be a list of indicator names, got list"):
        df.ta.strategy("momentum", exclude=["rsi", 7], cores=0)


def test_exclude_on_a_custom_strategy_raises():
    """It was popped and dropped, so it silently had no effect."""
    df = sample_frame(200)
    strategy = ta.Strategy("S", [{"kind": "rsi"}, {"kind": "sma", "length": 10}])
    with pytest.raises(ValueError, match=r"exclude does not apply to the custom Strategy 'S'"):
        df.ta.strategy(strategy, exclude=["rsi"], cores=0)


@pytest.mark.parametrize("retired", ["chunksize", "ordered"])
def test_retired_pool_keywords_warn_rather_than_pass_through(retired):
    """strategy() broadcasts unknown keywords, so these need an explicit check.

    Both were documented, so they get a deprecation cycle rather than simply
    disappearing into **kwargs and reaching the indicators unnoticed.
    """
    df = sample_frame(200)
    with pytest.warns(DeprecationWarning, match=rf"strategy\(\) {retired} is not used and has no effect"):
        df.ta.strategy("momentum", cores=0, **{retired: 4})
    assert "RSI_14" in df.columns  # ... and the run still happened


def test_non_strategy_argument_raises():
    df = sample_frame(200)
    with pytest.raises(TypeError, match=r"strategy\(\) expected a category name or a Strategy, got int"):
        df.ta.strategy(42)


@pytest.mark.parametrize("cores", [0, 2])
def test_indicator_reading_an_unknown_column_warns(cores):
    """A missing input column dropped the indicator silently, serially and in a worker."""
    df = sample_frame(200)
    strategy = ta.Strategy("Typo", [{"kind": "ema", "close": "NOT_A_COLUMN", "length": 5}])
    with pytest.warns(UserWarning, match=r"ema\(\) returned no result.*'NOT_A_COLUMN'"):
        df.ta.strategy(strategy, cores=cores)


def test_strategy_runs_inside_a_pool_worker():
    with get_context("spawn").Pool(1) as pool:
        columns = pool.map(_strategy_in_worker, [None])[0]
    assert columns > 5


def test_worker_error_names_the_indicator(executor):
    """An exception in a worker used to surface as a bare pickling traceback."""
    df = sample_frame(200)
    strategy = ta.Strategy("Bad", [{"kind": "sma", "length": 0}])
    with pytest.raises(RuntimeError, match=r"strategy\(\): sma\(\) raised in a worker process"):
        df.ta.strategy(strategy, executor=executor)
