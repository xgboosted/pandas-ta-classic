"""How strategy() executes: stages, serial/parallel parity, and argument checks.

The chunked Multiprocessing Pool that these replace decided correctness by
batching: a custom entry reading an earlier entry's column only saw it when
both happened to land in the same chunk, and produced no column otherwise,
without an error. The parity tests below run the same strategies serially and
on an Executor and demand identical frames.
"""

import os
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import cpu_count, get_context

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


def test_chained_column_resolves_case_insensitively(executor):
    """A stage boundary agrees with the lookup the indicator itself will do.

    _get_column() resolves a name the frame does not carry exactly through a
    case-insensitive exact match, so 'sma_10' reads SMA_10. _missing_column()
    goes through the same lookup: the entry still waits for its own stage,
    because nothing answers to 'sma_10' before SMA_10 is appended, and
    _worker_columns() then ships SMA_10 under the name the frame carries. A
    boundary decided on the literal string instead would leave the worker
    without the column.
    """
    entries = [{"kind": "sma", "length": 10}, {"kind": "ema", "close": "sma_10", "length": 5, "suffix": "CLR"}]

    serial = sample_frame()
    serial.ta.strategy(ta.Strategy("Chain", entries), cores=0)

    parallel = sample_frame()
    parallel.ta.strategy(ta.Strategy("Chain", entries), executor=executor)

    assert_same_frame(serial, parallel)
    assert "EMA_5_CLR" in parallel.columns
    assert parallel["EMA_5_CLR"].notna().any()


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


@pytest.mark.parametrize("mode", ["cores", "executor"])
def test_worker_error_keeps_the_type_it_had_serially(mode, executor):
    """The exception type is part of the contract, so opting into workers must not change it.

    This wrapped the cause in `RuntimeError: strategy(): sma() raised in a worker
    process`, so `except ValueError` stopped working the moment a caller set
    cores. The traceback still does not survive pickling; the indicator's name is
    in the message, plus a note where the interpreter has them.
    """
    df = sample_frame(200)
    strategy = ta.Strategy("Bad", [{"kind": "sma", "length": 0}])
    kwargs = {"executor": executor} if mode == "executor" else {"cores": 2}

    with pytest.raises(ValueError, match=r"sma\(\) length must be an integer > 0, got 0") as raised:
        df.ta.strategy(strategy, **kwargs)

    if hasattr(raised.value, "__notes__"):  # PEP 678, Python 3.11+
        assert any("raised by sma() in a worker process" in note for note in raised.value.__notes__)


def test_serial_and_worker_paths_raise_the_same_type():
    """The parity the test above asserts, stated directly against the serial path."""
    df = sample_frame(200)
    strategy = ta.Strategy("Bad", [{"kind": "sma", "length": 0}])
    with pytest.raises(ValueError) as serial:
        df.ta.strategy(strategy, cores=0)
    with pytest.raises(ValueError) as parallel:
        df.ta.strategy(strategy, cores=2)
    assert type(serial.value) is type(parallel.value)
    assert str(serial.value) == str(parallel.value)


def test_cores_kwarg_is_capped_at_cpu_count(monkeypatch):
    """`df.ta.cores` capped and the kwarg did not, so one knob had two limits."""
    seen: list[int] = []

    class _SpyExecutor(ProcessPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            seen.append(max_workers)
            super().__init__(1, **kwargs)

    monkeypatch.setattr(ta.core, "ProcessPoolExecutor", _SpyExecutor)
    sample_frame(200).ta.strategy(ta.Strategy("Cap", [{"kind": "sma", "length": 10}]), cores=10_000)
    assert seen == [cpu_count()]


def test_broken_pool_points_at_the_main_guard(monkeypatch):
    """A dead worker surfaced as BrokenProcessPool, which names no cause.

    The guard that catches the unguarded `__main__` raises in the *child*, whose
    stderr is gone in a notebook or a GUI process, so the parent was left with
    'terminated abruptly' and nothing to act on.
    """

    class _DeadPool:
        def submit(self, *_args, **_kwargs):
            future: Future = Future()
            future.set_exception(BrokenProcessPool("terminated abruptly"))
            return future

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(ta.core, "ProcessPoolExecutor", lambda *_a, **_k: _DeadPool())
    with pytest.raises(RuntimeError, match=r'wrap it in `if __name__ == "__main__":`') as raised:
        sample_frame(200).ta.strategy(ta.Strategy("Dead", [{"kind": "sma", "length": 10}]), cores=2)
    assert isinstance(raised.value.__cause__, BrokenProcessPool)


@pytest.mark.parametrize("mode", ["serial", "cores", "executor"])
def test_no_result_warning_blames_the_callers_line(mode, executor):
    """stacklevel was fixed at 4, but the parallel path is one frame deeper.

    Both parallel modes are checked, not just one: they reach the warning through
    the same _run_stages -> _run_stage frames, so a depth tuned to one of them
    should hold for the other, and this is what says so.
    """
    strategy = ta.Strategy("Typo", [{"kind": "ema", "close": "NOT_A_COLUMN", "length": 5}])
    kwargs = {"executor": executor} if mode == "executor" else {"cores": 2 if mode == "cores" else 0}
    with pytest.warns(UserWarning, match=r"returned no result") as caught:
        sample_frame(200).ta.strategy(strategy, **kwargs)
    assert [os.path.basename(w.filename) for w in caught] == [os.path.basename(__file__)]


def test_executor_path_also_sets_the_recursion_guard(executor, monkeypatch):
    """The guard lived in the cores>0 branch only.

    An unguarded script that builds its own pool and passes `executor=` recurses
    exactly like one that lets strategy() open the pool, because
    ProcessPoolExecutor spawns on the first submit -- inside strategy() either way.
    """
    seen: list[str | None] = []
    original = ta.core.AnalysisIndicators._run_stages

    def spy(self, *args, **kwargs):
        seen.append(os.environ.get(ta.core._STRATEGY_GUARD_ENV))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ta.core.AnalysisIndicators, "_run_stages", spy)
    sample_frame(200).ta.strategy(ta.Strategy("Guard", [{"kind": "sma", "length": 10}]), executor=executor)
    assert seen == [str(os.getpid())]
    assert os.environ.get(ta.core._STRATEGY_GUARD_ENV) is None


def test_a_finished_call_does_not_clear_another_threads_guard(executor, monkeypatch):
    """The `finally` popped unconditionally, so the first thread to finish disarmed the second.

    Children spawned after that point inherited no marker and would have recursed.
    """
    import threading

    both_inside = threading.Barrier(2, timeout=30)
    first_has_left = threading.Event()
    marker_after = []
    original = ta.core.AnalysisIndicators._run_stages

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        both_inside.wait()
        if self._df.attrs.get("_role") == "second":
            # The other call has returned and run its finally by now.
            first_has_left.wait(timeout=30)
            marker_after.append(os.environ.get(ta.core._STRATEGY_GUARD_ENV))
        return result

    monkeypatch.setattr(ta.core.AnalysisIndicators, "_run_stages", spy)

    def run(role: str) -> None:
        df = sample_frame(200)
        df.attrs["_role"] = role
        df.ta.strategy(ta.Strategy("T", [{"kind": "sma", "length": 10}]), executor=executor)
        if role == "first":
            first_has_left.set()

    threads = [threading.Thread(target=run, args=(role,)) for role in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=45)

    assert marker_after == [str(os.getpid())], "the second call lost its guard when the first finished"
    assert os.environ.get(ta.core._STRATEGY_GUARD_ENV) is None
