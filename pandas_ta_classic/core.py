import logging
import os
from collections.abc import Hashable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import cpu_count, current_process, get_context
from numbers import Integral
from time import perf_counter
from typing import Any
from warnings import simplefilter, warn

import pandas as pd
from pandas.core.base import PandasObject

from pandas_ta_classic._indicator_loader import _COLUMN_KWARG_KEYS, _DEFAULT_COLUMN_NAMES, _find_indicator_func, _make_ta_wrapper
from pandas_ta_classic._meta import _MATH_ALIASES, EXCHANGE_TZ, Category, Imports, version
from pandas_ta_classic.utils import final_time, get_time, is_datetime_ordered, to_utc, total_time
from pandas_ta_classic.utils._core import _bool_param, _pos_int
from pandas_ta_classic.utils._time import TIME_RANGE_UNITS

logger = logging.getLogger(__name__)

# Set while strategy() owns a process pool it created itself.  A spawned child
# re-imports the caller's __main__, so an unguarded script that calls strategy()
# at module level would recurse until the machine stalls.  The variable is
# inherited by the children, so seeing it means we are that recursion.
_STRATEGY_GUARD_ENV = "_PANDAS_TA_CLASSIC_STRATEGY_PID"

_MAIN_GUARD_HINT = (
    "df.ta.strategy() started worker processes from a script that does not guard "
    "its top level. Wrap the calling code in `if __name__ == \"__main__\":`, or run "
    "serially with df.ta.cores = 0."
)


# Strategy DataClass
@dataclass
class Strategy:
    """Strategy DataClass
    A way to name and group your favorite indicators

    Args:
        name (str): Some short memorable string.  Note: Case-insensitive "All" is reserved.
        ta (list of dicts): A list of dicts containing keyword arguments where "kind" is the indicator.
        description (str): A more detailed description of what the Strategy tries to capture. Default: None
        created (str): At datetime string of when it was created. Default: Automatically generated. *Subject to change*

    Example TA:
    ta = [
        {"kind": "sma", "length": 200},
        {"kind": "sma", "close": "volume", "length": 50},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
    """

    name: str  # = None # Required.
    ta: list | None = field(default_factory=list)  # Required. None means every indicator.
    # Helpful. More descriptive version or notes or w/e.
    description: str = "TA Description"
    # Optional. Gets Exchange Time and Local Time execution time
    created: str | None = field(default_factory=lambda: get_time(to_string=True))

    def __post_init__(self):
        required_args = ["[X] Strategy requires the following argument(s):"]

        name_is_str = isinstance(self.name, str)
        ta_is_list = isinstance(self.ta, list)

        if self.name is None or not name_is_str:
            required_args.append(' - name. Must be a string. Example: "My TA". Note: "all" is reserved.')

        if self.ta is not None and not ta_is_list:
            s = " - ta. Format is a list of dicts. Example: [{'kind': 'sma', 'length': 10}]"
            s += "\n       Check the indicator for the correct arguments if you receive this error."
            required_args.append(s)

        if len(required_args) > 1:
            raise ValueError("\n".join(required_args))

    def total_ta(self):
        return len(self.ta) if self.ta is not None else 0


# All Default Strategy
AllStrategy = Strategy(
    name="All",
    description="All the indicators with their default settings. Pandas TA default.",
    ta=None,
)

# Default (Example) Strategy.
CommonStrategy = Strategy(
    name="Common Price and Volume SMAs",
    description="Common Price SMAs: 10, 20, 50, 200 and Volume SMA: 20.",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOL"},
    ],
)


def _append_dataframe(df, result, kwargs):
    """Append a DataFrame *result* to *df*, honouring optional col_names in *kwargs*."""
    if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple):
        if len(kwargs["col_names"]) >= len(result.columns):
            for col, ind_name in zip(result.columns, kwargs["col_names"]):
                df[ind_name] = result.loc[:, col]
        else:
            logger.error(f"Not enough col_names were specified: got {len(kwargs['col_names'])}, expected {len(result.columns)}.")
            return
    else:
        for i, column in enumerate(result.columns):
            df[column] = result.iloc[:, i]


def _strategy_params(ind: dict) -> tuple:
    """Positional arguments of a custom Strategy entry.

    A ``params`` value that is not a tuple (``[10]``, ``10``) was silently
    replaced by ``()``, so the indicator ran with its defaults.
    """
    params = ind.get("params", ())
    if not isinstance(params, tuple):
        raise TypeError(f"Strategy entry {ind.get('kind')!r}: params must be a tuple, got {type(params).__name__} {params!r}")
    return params


def _executor_workers(executor: Executor) -> int:
    """How many tasks *executor* can run at once.

    Executor exposes no public width, but every stdlib implementation carries
    '_max_workers'. A third-party one that does not gets one group per worker
    at cpu_count(), which is only a batching hint: correctness does not depend
    on the number being right.
    """
    workers = getattr(executor, "_max_workers", None)
    return workers if isinstance(workers, int) and workers > 0 else cpu_count()


def _run_task_group(df: pd.DataFrame, group: list[tuple]) -> list[tuple]:
    """Run a group of strategy tasks and return (order, result, error) triples.

    Runs in a worker process.  The tasks of a group are independent by
    construction (see AnalysisIndicators._next_stage), so nothing here appends
    to *df*; the parent appends in task order.
    """
    # Chain mode would make every indicator return the whole frame.
    df.attrs.pop("_ta_chain", None)
    accessor = df.ta
    out: list[tuple[int, Any, Exception | None]] = []
    for order, kind, params, kwargs in group:
        try:
            out.append((order, getattr(accessor, kind)(*params, **{**kwargs, "append": False}), None))
        except Exception as exc:  # noqa: BLE001 - re-raised in the parent with the indicator name
            out.append((order, None, exc))
    return out


# Pandas TA - DataFrame Analysis Indicators
@pd.api.extensions.register_dataframe_accessor("ta")
class AnalysisIndicators(PandasObject):
    """
    This Pandas Extension is named 'ta' for Technical Analysis. In other words,
    it is a Numerical Time Series Feature Generator where the Time Series data
    is biased towards Financial Market data; typical data includes columns
    named :"open", "high", "low", "close", "volume".

    This TA Library hopefully allows you to apply familiar and unique Technical
    Analysis Indicators easily with the DataFrame Extension named 'ta'. Even
    though 'ta' is a Pandas DataFrame Extension, you can still call Technical
    Analysis indicators individually if you are more comfortable with that
    approach or it allows you to easily and automatically apply the indicators
    with the strategy method. See: help(ta.strategy).

    By default, the 'ta' extension uses lower case column names: open, high,
    low, close, and volume. You can override the defaults by providing the it's
    replacement name when calling the indicator. For example, to call the
    indicator hl2().

    With 'default' columns: open, high, low, close, and volume.
    >>> df.ta.hl2()
    >>> df.ta(kind="hl2")

    With DataFrame columns: Open, High, Low, Close, and Volume.
    >>> df.ta.hl2(high="High", low="Low")
    >>> df.ta(kind="hl2", high="High", low="Low")

    If you do not want to use a DataFrame Extension, just call it normally.
    >>> sma10 = ta.sma(df["Close"]) # Default length=10
    >>> sma50 = ta.sma(df["Close"], length=50)
    >>> ichimoku = ta.ichimoku(df["High"], df["Low"], df["Close"], as_dataframe=True)

    Args:
        kind (str, optional): Default: None. Kind is the 'name' of the indicator.
            It converts kind to lowercase before calling.
        timed (bool, optional): Default: False. Curious about the execution
            speed?
        kwargs: Extension specific modifiers.
            append (bool, optional): Default: False. When True, it appends the
            resultant column(s) to the DataFrame.

    Returns:
        Most Indicators will return a Pandas Series. Others like MACD, BBANDS,
        KC, et al will return a Pandas DataFrame. Ichimoku returns a single
        DataFrame for the known period; pass ``append_span=True`` to also get
        the forward-looking Span rows.

    Let's get started!

    1. Loading the 'ta' module:
    >>> import pandas as pd
    >>> import ta as ta

    2. Load some data:
    >>> df = pd.read_csv("AAPL.csv", index_col="date", parse_dates=True)

    3. Help!
    3a. General Help:
    >>> help(df.ta)
    >>> df.ta()
    3b. Indicator Help:
    >>> help(ta.apo)
    3c. Indicator Extension Help:
    >>> help(df.ta.apo)

    4. Ways of calling an indicator.
    4a. Standard: Calling just the APO indicator without "ta" DataFrame extension.
    >>> ta.apo(df["close"])
    4b. DataFrame Extension: Calling just the APO indicator with "ta" DataFrame extension.
    >>> df.ta.apo()
    4c. DataFrame Extension (kind): Calling APO using 'kind'
    >>> df.ta(kind="apo")
    4d. Strategy:
    >>> df.ta.strategy("All") # Default
    >>> df.ta.strategy(ta.Strategy("My Strat", ta=[{"kind": "apo"}])) # Custom

    5. Working with kwargs
    5a. Append the result to the working df.
    >>> df.ta.apo(append=True)
    5b. Timing an indicator.
    >>> apo = df.ta(kind="apo", timed=True)
    >>> print(apo.timed)
    """

    _adjusted = None
    _cores = 0
    _df = pd.DataFrame()
    _exchange = "NYSE"
    _time_range = "years"

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._df = pandas_obj
        # pandas 3 dropped accessor caching: 'df.ta' builds a new instance on
        # every access, so settable state lives in df.attrs (like '_ta_chain')
        # instead of on the instance, which would be discarded immediately.
        # Nothing is written here: merely accessing df.ta must not mutate the
        # caller's DataFrame.

    @staticmethod
    def _validate(obj: tuple[pd.DataFrame, pd.Series]):
        if not isinstance(obj, pd.DataFrame) and not isinstance(obj, pd.Series):
            # pandas accessor contract: _validate raises AttributeError
            raise AttributeError("[X] Must be either a Pandas Series or DataFrame.")  # noqa: TRY004

    # DataFrame Behavioral Methods
    def __call__(
        self,
        kind: str | None = None,
        timed: bool = False,
        show_version: bool = False,
        **kwargs,
    ):
        show_version = _bool_param(show_version, False, "show_version")
        version = kwargs.pop("version", None)
        if version is not None:
            warn(
                "df.ta(version=...) is deprecated and will be removed in the next breaking release; use show_version=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            version_val = _bool_param(version, False, "version")
            show_version = show_version or version_val
        if show_version:
            logger.info(f"Pandas TA - Technical Analysis Indicators - v{self.version}")
        if isinstance(kind, str):
            kind = kind.lower()
            fn = getattr(self, kind, None)
            if fn is None:
                logger.error("Indicator '%s' not found.", kind)
                self.help()
                return None
            if not callable(fn):
                logger.error("'%s' is not a callable indicator.", kind)
                return None

            if timed:
                stime = perf_counter()

            # Run the indicator
            result = fn(**kwargs)
            self._df.attrs["_ta_last_run"] = get_time(self.exchange, to_string=True)  # Save when it completed it's run

            if timed:
                if result is not None:
                    result.timed = final_time(stime)
                    logger.info("%s: %s", kind, result.timed)
                else:
                    logger.warning("%s: returned None, timed run produced no result", kind)

            return result
        self.help()

    # Public Get/Set DataFrame Properties
    @property
    def adjusted(self) -> str | None:
        """property: df.ta.adjusted"""
        return self._df.attrs.get("_ta_adjusted", self._adjusted)

    @adjusted.setter
    def adjusted(self, value: str) -> None:
        """property: df.ta.adjusted = 'adj_close'"""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"df.ta.adjusted must be a column name or None, got {value!r}")
        self._df.attrs["_ta_adjusted"] = value

    @property
    def cores(self) -> int:
        """Returns the categories."""
        return self._df.attrs.get("_ta_cores", self._cores)

    @cores.setter
    def cores(self, value: int) -> None:
        """property: df.ta.cores = integer (0, the default, runs serially; capped at cpu_count(); None resets)"""
        cpus = cpu_count()
        if value is None:
            self._df.attrs["_ta_cores"] = self._cores
            return
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            # -1, 1.0 and "2" used to become cpu_count() and switch multiprocessing on
            raise ValueError(f"df.ta.cores must be an integer >= 0 or None, got {value!r}")
        self._df.attrs["_ta_cores"] = min(int(value), cpus)

    @property
    def exchange(self) -> str:
        """Returns the current Exchange. Default: "NYSE"."""
        return self._df.attrs.get("_ta_exchange", self._exchange)

    @exchange.setter
    def exchange(self, value: str) -> None:
        """property: df.ta.exchange = "LSE" (None resets to NYSE)"""
        if value is None:
            self._df.attrs.pop("_ta_exchange", None)
            return
        if not isinstance(value, str) or value not in EXCHANGE_TZ:
            # an unknown exchange used to be ignored, leaving the previous one in place
            raise ValueError(f"df.ta.exchange must be one of {sorted(EXCHANGE_TZ)} or None, got {value!r}")
        self._df.attrs["_ta_exchange"] = value

    @property
    def last_run(self) -> str | None:
        """Returns when df.ta(kind=...) or df.ta.strategy() last ran on the DataFrame, or None."""
        return self._df.attrs.get("_ta_last_run")

    # Public Get DataFrame Properties
    @property
    def categories(self) -> list[str]:
        """Returns the categories."""
        return list(Category.keys())

    @property
    def datetime_ordered(self) -> bool:
        """Returns True if the index is a datetime and ordered."""
        hasdf = hasattr(self, "_df")
        if hasdf:
            return is_datetime_ordered(self._df)
        return hasdf

    @property
    def reverse(self) -> pd.DataFrame:
        """Reverses the DataFrame. Simply: df.iloc[::-1]"""
        return self._df.iloc[::-1]

    @property
    def time_range(self) -> float:
        """Returns the time ranges of the DataFrame as a float. Default is in "years". help(ta.toal_time)"""
        return total_time(self._df, self._df.attrs.get("_ta_time_range", self._time_range))

    @time_range.setter
    def time_range(self, value: str) -> None:
        """property: df.ta.time_range = "years" (Default; None resets)"""
        if value is not None and value not in TIME_RANGE_UNITS:
            # an unknown unit used to be stored and then computed as years
            raise ValueError(f"df.ta.time_range must be one of {list(TIME_RANGE_UNITS)} or None, got {value!r}")
        self._df.attrs["_ta_time_range"] = "years" if value is None else value

    @property
    def to_utc(self) -> None:
        """Sets the DataFrame index to UTC format"""
        self._df = to_utc(self._df)

    @property
    def version(self) -> str:
        """Returns the version."""
        return version

    # Fluent API chaining (Issue #36)
    def chain(self, append: bool = True):
        """Activate fluent chaining mode.

        When chain mode is active, every indicator call auto-appends its result
        to the DataFrame and returns the DataFrame itself (which has ``.ta``),
        so you can chain multiple indicators without repeating ``df.ta``::

            df.ta.chain().sma(10).ta.rsi(14).ta.macd()

        Args:
            append (bool): When True (default), each indicator is appended to
                the DataFrame.

        Returns:
            AnalysisIndicators: self (the accessor) with chain mode active.
        """
        self._df.attrs["_ta_chain"] = True
        self._df.attrs["_ta_chain_append"] = append
        return self

    def unchain(self):
        """Deactivate fluent chaining mode.

        Returns:
            pd.DataFrame: The working DataFrame (so ``.ta`` is available for
            non-chained calls).
        """
        self._df.attrs.pop("_ta_chain", None)
        self._df.attrs.pop("_ta_chain_append", None)
        return self._df

    # Private DataFrame Methods
    def _add_prefix_suffix(self, result=None, **kwargs) -> None:
        """Add prefix and/or suffix to the result columns"""
        if result is None:
            return
        prefix = suffix = ""
        delimiter = kwargs.setdefault("delimiter", "_")

        if "prefix" in kwargs:
            prefix = f"{kwargs['prefix']}{delimiter}"
        if "suffix" in kwargs:
            suffix = f"{delimiter}{kwargs['suffix']}"

        if isinstance(result, pd.Series):
            result.name = prefix + result.name + suffix
        else:
            result.columns = [prefix + column + suffix for column in result.columns]

    def _append(self, result=None, **kwargs) -> None:
        """Appends a Pandas Series or DataFrame columns to self._df."""
        if not kwargs.get("append"):
            return
        df = self._df
        if df is None or result is None:
            return
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        if "col_names" in kwargs and not isinstance(kwargs["col_names"], tuple):
            kwargs["col_names"] = (kwargs["col_names"],)
        if isinstance(result, pd.DataFrame):
            _append_dataframe(df, result, kwargs)
        else:
            ind_name = kwargs["col_names"][0] if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple) else result.name
            df[ind_name] = result

    def _default_column(self, name: str) -> str:
        """The column an indicator reads when the caller names none.

        'close' resolves to df.ta.adjusted when that is set; every other
        column name is returned unchanged.
        """
        if name == "close" and self.adjusted is not None:
            return self.adjusted
        return name

    def _get_column(self, series):
        """Attempts to get the correct series or 'column' and return it."""
        df = self._df
        if df is None:
            return

        # Explicitly passing a pd.Series to override default.
        if isinstance(series, pd.Series):
            return series
        # Apply default if no series nor a default.
        if series is None:
            return df[self.adjusted] if self.adjusted is not None else None
        # Ok.  So it's a str.
        if isinstance(series, str):
            # Return the df column since it's in there.
            if series in df.columns:
                return df[series]
            # Attempt to match the 'series' because it was likely
            # misspelled.
            matches = df.columns.str.match(series, case=False)
            match = [i for i, x in enumerate(matches) if x]
            if len(match):
                return df.iloc[:, match[0]]
            cols = ", ".join(list(df.columns))
            logger.warning(f"[X] Column '{series}' not found. Available columns: {cols}")
            return None
        # Anything else (a numpy array, a list, a DataFrame) is passed through
        # unchanged so verify_series() can warn about it; returning None here
        # made df.ta.sma(close=df.close.values) a silent no-op.
        return series

    def _matching_column(self, name: str) -> Hashable | None:
        """The column '_get_column' would resolve *name* to, or None.

        Mirrors _get_column()'s lookup order: the exact name first, then the
        case-insensitive prefix match.
        """
        columns = self._df.columns
        if name in columns:
            return name
        # Index.str is unavailable on a non-string column Index; _get_column
        # cannot resolve a prefix match there either, so there is nothing
        # further to look up.
        if columns.inferred_type != "string":
            return None
        matches = [i for i, hit in enumerate(columns.str.match(name, case=False)) if hit]
        return columns[matches[0]] if matches else None

    def _worker_columns(self, kwarg_sources: list[dict]) -> list:
        """The columns a Multiprocessing worker can actually reach for.

        An indicator only ever sees the default OHLCV columns, whatever a
        ``close="my_col"``-style kwarg points at, and ``df.ta.adjusted``.
        Everything else — a previous strategy() run's output, unrelated user
        columns — would be pickled onto the Pool task pipe for nothing.

        Args:
            kwarg_sources (list of dict): Every kwargs mapping that will reach
                an indicator call, i.e. the shared strategy kwargs plus each
                per-indicator dict of a Custom Strategy.

        Returns:
            list: Column labels, in DataFrame order.
        """
        requested = set(_DEFAULT_COLUMN_NAMES)
        if self.adjusted is not None:
            requested.add(self.adjusted)
        for source in kwarg_sources:
            for key in _COLUMN_KWARG_KEYS:
                value = source.get(key)
                if isinstance(value, str):
                    requested.add(value)

        resolved = {column for column in (self._matching_column(name) for name in requested) if column is not None}
        return [column for column in self._df.columns if column in resolved]

    def _indicators_by_category(self, name: str) -> list | None:
        """Returns indicators by Categorical name."""
        return Category[name] if name in self.categories else None

    def _missing_column(self, task_kwargs: dict) -> str | None:
        """The column *task_kwargs* points an indicator at that does not exist yet.

        A strategy entry such as ``{"kind": "ema", "close": "SMA_10"}`` can only
        run once ``SMA_10`` is on the frame.

        A name that resolves only through _get_column()'s case-insensitive prefix
        match counts as missing too.  The match is a fallback for a misspelled
        column, and an unrelated column can satisfy it: ``close="MA"`` resolves
        to ``MACD_12_26_9``.  Treating that as present would put the entry in the
        same stage as the entry producing the exact name, and the worker -- whose
        frame does not carry that name yet -- would silently read the other
        column.  Holding it back costs a stage; the exact name wins once it is
        appended.
        """
        for key in _COLUMN_KWARG_KEYS:
            value = task_kwargs.get(key)
            if isinstance(value, str) and self._matching_column(value) != value:
                return value
        return None

    def _next_stage(self, remaining: list[tuple]) -> list[tuple]:
        """Take the next run of mutually independent tasks off *remaining*.

        A task that reads a column the frame does not have yet starts a new
        stage, so everything it depends on has been appended by the time it
        runs.  Within a stage the order does not matter, which is what makes
        parallel execution safe: the previous chunked Pool ran a producer and
        its consumer in different workers unless they happened to land in the
        same chunk, and the consumer then silently produced no column at all.

        Called once per stage rather than up front, because where the next
        boundary falls depends on the columns the finished stages added.

        A chain of entries that each read the previous one's output is one task
        per stage, which is what it is -- they cannot run at the same time.  An
        entry naming a column nothing ever produces costs a stage of its own
        too; it also warns (see _warn_no_result), so the cause is visible.
        """
        stage = [remaining.pop(0)]
        while remaining and self._missing_column(remaining[0][3]) is None:
            stage.append(remaining.pop(0))
        return stage

    def _warn_no_result(self, kind: str, task_kwargs: dict) -> None:
        """Report an indicator that ran but added no column.

        Both halves of this used to be invisible: _get_column() logged its
        "Column not found" warning through the package NullHandler -- in a Pool
        worker, where even a configured handler could not show it -- and
        _post_process() then dropped the result.  A chained custom strategy
        quietly produced fewer columns than it was given entries.
        """
        missing = self._missing_column(task_kwargs)
        detail = f" It reads {missing!r}, which the DataFrame does not have." if missing else ""
        warn(f"strategy(): {kind}() returned no result, so no column was added.{detail}", UserWarning, stacklevel=4)

    def _run_serially(self, tasks: list[tuple], verbose: bool) -> None:
        """Run every task in order, appending each result before the next runs."""
        iterator: Any = tasks
        if Imports["tqdm"] and verbose:
            from tqdm import tqdm  # type: ignore[import-untyped]  # optional; ships no stubs

            iterator = tqdm(tasks, "[i] Progress")
        for _order, kind, params, task_kwargs in iterator:
            if getattr(self, kind)(*params, **task_kwargs) is None:
                self._warn_no_result(kind, task_kwargs)

    def _run_stages(self, tasks: list[tuple], executor: Executor, workers: int, verbose: bool) -> None:
        """Run *tasks* stage by stage on *executor*, appending between stages."""
        remaining = list(tasks)
        stage_number = 0
        while remaining:
            stage = self._next_stage(remaining)
            stage_number += 1
            if verbose:
                logger.info(f"Stage {stage_number}: {len(stage)} indicators over {workers} workers.")
            self._run_stage(stage, executor, workers)

    def _run_stage(self, stage: list[tuple], executor: Executor, workers: int) -> None:
        """Run one stage of independent tasks and append the results in order."""
        groups = min(len(stage), workers)
        # Send only the columns an indicator in this stage can reach.  The frame
        # grows with every appended result: after one strategy("momentum") run
        # on 100,000 rows it pickles to 79.4 MB against 4.6 MB for the five
        # columns the next stage can actually read.  df.ta settings travel with
        # it, because pandas carries DataFrame.attrs through the selection.
        slim = self._df[self._worker_columns([task_kwargs for _, _, _, task_kwargs in stage])].copy()

        # Round robin, so one group per worker holds a spread of cheap and
        # expensive indicators, and the frame is pickled once per group.
        futures = [executor.submit(_run_task_group, slim, stage[start::groups]) for start in range(groups)]
        finished: dict[int, tuple] = {}
        for future in futures:
            finished.update({order: (result, error) for order, result, error in future.result()})

        # Append by task order, not by the order the workers finished.
        for order, kind, _params, task_kwargs in stage:
            result, error = finished[order]
            if error is not None:
                raise RuntimeError(f"strategy(): {kind}() raised in a worker process") from error
            if result is None:
                self._warn_no_result(kind, task_kwargs)
                continue
            self._append(result, **task_kwargs)

    def _post_process(self, result, **kwargs):
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        * In chain mode, auto-appends and returns the DataFrame for fluent chaining.
        """
        verbose = _bool_param(kwargs.pop("verbose", None), False, "verbose")
        chain_mode = self._df.attrs.get("_ta_chain", False)

        if not isinstance(result, (pd.Series, pd.DataFrame)):
            # Returning the whole DataFrame here made "no result" indistinguishable
            # from a real one without an identity check, and df.ta.sma(close="typo")
            # handed back the frame it was called on. Chain mode still returns the
            # frame, because there the frame *is* the result.
            if verbose:
                logger.error("The result was not a Series or DataFrame.")
            return self._df if chain_mode else None
        # Append only specific columns to the dataframe (via
        # 'col_numbers':(0,1,3) for example)
        result = (
            result.iloc[:, [int(n) for n in kwargs["col_numbers"]]]
            if isinstance(result, pd.DataFrame) and "col_numbers" in kwargs and kwargs["col_numbers"] is not None
            else result
        )
        # Add prefix/suffix and append to the dataframe
        self._add_prefix_suffix(result=result, **kwargs)
        # In chain mode, auto-append results to the DataFrame
        if chain_mode:
            kwargs["append"] = self._df.attrs.get("_ta_chain_append", True)
        self._append(result=result, **kwargs)

        # In chain mode, return the DataFrame (which has .ta) for fluent chaining
        if chain_mode:
            return self._df
        return result

    def _strategy_mode(self, *args) -> tuple:
        """Helper method to determine the mode and name of the strategy. Returns tuple: (name:str, mode:dict)"""
        if len(args) == 0:
            return "All", {"all": True, "category": False, "custom": False}
        return self._resolve_strategy_args(args[0])

    def _resolve_strategy_args(self, arg) -> tuple:
        """Resolve (name, mode) from a single strategy argument (str or Strategy)."""
        name = "All"
        mode = {"all": False, "category": False, "custom": False}
        if isinstance(arg, str):
            if arg.lower() == "all":
                mode["all"] = True
            elif arg.lower() in self.categories:
                name, mode["category"] = arg, True
            else:
                # A Strategy's *name* never selected anything here, so
                # df.ta.strategy("CommonStrategy") logged an invisible error and
                # added no columns.  Pass the Strategy object itself.
                categories = ", ".join(sorted(self.categories))
                raise ValueError(f"strategy() got {arg!r}, which is neither 'all' nor a category ({categories}). Pass a Strategy to run a named one.")
            return name, mode
        if isinstance(arg, Strategy):
            strategy_ = arg
            if strategy_.ta is None or strategy_.name.lower() == "all":
                mode["all"] = True
            elif strategy_.name.lower() in self.categories:
                name, mode["category"] = strategy_.name, True
            else:
                name, mode["custom"] = strategy_.name, True
            return name, mode
        raise TypeError(f"strategy() expected a category name or a Strategy, got {type(arg).__name__}")

    # Public DataFrame Methods
    def indicators(self, **kwargs):
        """List of Indicators

        kwargs:
            as_list (bool, optional): When True, it returns a list of the
                indicators. Default: False.
            exclude (list, optional): The passed in list will be excluded
                from the indicators list. Default: None.

        Returns:
            Prints the list of indicators. If as_list=True, then a list.
        """
        as_list = kwargs.setdefault("as_list", False)
        # Public non-indicator methods
        helper_methods = [
            "chain",
            "indicators",
            "strategy",
            "unchain",
        ]
        # Public df.ta.properties
        ta_properties = [
            "adjusted",
            "categories",
            "cores",
            "datetime_ordered",
            "exchange",
            "last_run",
            "reverse",
            "time_range",
            "to_utc",
            "version",
        ]

        # Build indicator list from Category (works with lazy __getattr__)
        # Also include explicitly-defined methods (math operators) that are not
        # in Category but are available on the accessor.
        from pandas_ta_classic._meta import Category as _Category

        _category_indicators = [ind for inds in _Category.values() for ind in inds]
        ta_indicators = sorted(set(_category_indicators))

        # Add Pandas TA methods and properties to be removed
        removed = helper_methods + ta_properties

        # Add user excluded methods to be removed
        user_excluded = kwargs.setdefault("exclude", [])
        if isinstance(user_excluded, list) and len(user_excluded) > 0:
            removed += user_excluded

        # Remove the unwanted indicators (only if present)
        for x in removed:
            if x in ta_indicators:
                ta_indicators.remove(x)

        # If as a list, immediately return
        if as_list:
            return ta_indicators

        total_indicators = len(ta_indicators)
        header = f"Pandas TA - Technical Analysis Indicators - v{self.version}"
        from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS

        s = f"{header}\nTotal Indicators & Utilities: {total_indicators + len(ALL_PATTERNS)}\n"
        if total_indicators > 0:
            print(f"{s}Abbreviations:\n    {', '.join(ta_indicators)}\n\nCandle Patterns:\n    {', '.join(ALL_PATTERNS)}")
        else:
            print(s)

    def strategy(self, *args, **kwargs):
        """Strategy Method

        An experimental method that by default runs all applicable indicators.
        Future implementations will allow more specific indicator generation
        with possibly as json, yaml config file or an sqlite3 table.


        Runs serially by default. Parallel execution is opt-in and worth it from
        roughly 100,000 rows per frame upwards; below that, starting worker
        processes costs more than the indicators do. See docs/strategies.rst.

        Kwargs:
            chunksize (int): Deprecated and unused: it sized the batches of the
                Multiprocessing Pool that stages replace. Passing it emits a
                DeprecationWarning. Default: None
            cores (int): Run this call on that many worker processes, 0 for
                serially. Default: df.ta.cores, itself 0 unless set.
            executor (concurrent.futures.Executor): Run this call on a pool the
                caller owns and reuses, which avoids paying for process start-up
                per call. Takes precedence over 'cores'. Default: None
            exclude (list): List of indicator names to exclude. Some are
                excluded by default for various reasons; they require additional
                sources, performance (td_seq), not a ohlcv chart (vp) etc.
            name (str): Select all indicators or indicators by
                Category such as: "candles", "cycles", "momentum", "overlap",
                "performance", "statistics", "trend", "volatility", "volume", or
                "all". Default: "all"
            ordered (bool): Deprecated and unused: results are always appended
                in the order the entries are listed. Passing it emits a
                DeprecationWarning. Default: None
            timed (bool): Show the process time of the strategy().
                Default: False
            verbose (bool): Provide some additional insight on the progress of
                the strategy() execution. Default: False
        """
        # If True, it returns the resultant DataFrame. Default: False
        returns = _bool_param(kwargs.pop("returns", None), False, "returns")
        # Ensure indicators are appended to the DataFrame
        kwargs["append"] = True
        executor = kwargs.pop("executor", None)
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError(f"strategy() executor must be a concurrent.futures.Executor or None, got {type(executor).__name__}")
        # 'cores' used to be read off the accessor only; the kwarg was accepted
        # and dropped, so strategy(cores=0) still started a pool.
        cores = _pos_int(kwargs.pop("cores", None), self.cores, "cores", gt=None, ge=0)
        # strategy() broadcasts unknown keywords to the indicators, so a
        # keyword that lost its meaning would otherwise pass through in silence.
        for retired in ("chunksize", "ordered"):
            if kwargs.pop(retired, None) is not None:
                warn(
                    f"strategy() {retired} is not used and has no effect; it is deprecated and will be "
                    "removed in the next breaking release. Remove the argument.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Initialize
        initial_column_count = len(self._df.columns)
        excluded = [
            "above",
            "above_value",
            "below",
            "below_value",
            "cross",
            "cross_value",
            # "data", # reserved
            "long_run",
            "mavp",  # Requires a per-bar 'periods' Series
            "short_run",
            "td_seq",  # Performance exclusion
            "tsignals",
            "vp",
            "xsignals",
        ]
        # These need an argument the frame cannot supply, and now raise without
        # it rather than returning nothing.  Leave them in when the caller
        # broadcasts one, so df.ta.strategy("all", benchmark=other) keeps
        # computing beta and correl as it did.
        if "benchmark" not in kwargs:
            excluded += ["beta", "correl"]
        if "name" not in kwargs:
            excluded.append("ma")  # a dispatcher: ma() with no name lists the available MAs

        # Get the Strategy Name and mode
        name, mode = self._strategy_mode(*args)

        # If All or a Category, exclude user list if any
        if not isinstance(self._df.index, pd.DatetimeIndex):
            excluded.append("vwap")  # anchors by calendar period; raises without a DatetimeIndex
        user_excluded = kwargs.pop("exclude", [])
        if mode["all"] or mode["category"]:
            excluded += user_excluded

        # Collect the indicators, remove excluded or include kwarg["append"].
        # Work on copies: `Category` lists and the caller's Strategy.ta are shared
        # objects, and removing from them would leak into later runs.
        ta: list
        if mode["category"]:
            ta = [x for x in Category[name.lower()] if x not in excluded]
        elif mode["custom"]:
            # custom mode implies a list: _resolve_strategy_args routes ta=None to "all"
            ta = [{**kwds, "append": True} for kwds in args[0].ta]
        else:  # mode["all"]; _resolve_strategy_args() raises on anything else
            ta = self.indicators(as_list=True, exclude=excluded)

        verbose = _bool_param(kwargs.pop("verbose", None), False, "verbose")
        if verbose:
            logger.info(f"Strategy: {name}\nIndicator arguments: {kwargs}")
            if mode["all"] or mode["category"]:
                excluded_str = ", ".join(excluded)
                logger.info(f"Excluded[{len(excluded)}]: {excluded_str}")

        timed = _bool_param(kwargs.pop("timed", None), False, "timed")

        # The plan: one (order, kind, params, kwargs) task per indicator, in the
        # order the caller asked for. Pure data -- nothing has run yet.
        if mode["custom"]:
            tasks = [(i, ind["kind"], _strategy_params(ind), {**ind, **kwargs}) for i, ind in enumerate(ta)]
        else:
            tasks = [(i, ind, (), dict(kwargs)) for i, ind in enumerate(ta)]

        # A daemonic worker cannot start children of its own ('daemonic
        # processes are not allowed to have children'), so a strategy() inside
        # someone else's pool runs serially instead of failing.
        if (executor is not None or cores > 0) and current_process().daemon:
            if verbose:
                logger.info("Running serially: strategy() was called inside a worker process.")
            executor, cores = None, 0

        if timed:
            stime = perf_counter()

        if executor is not None:
            self._run_stages(tasks, executor, _executor_workers(executor), verbose)
        elif cores > 0:
            # Compare the pid, do not just test for the variable: os.environ is
            # process-global, so a second thread calling strategy(cores=...)
            # would otherwise be told to guard its __main__.  A spawned child
            # inherits the *parent's* pid here, which never matches its own.
            guard_owner = os.environ.get(_STRATEGY_GUARD_ENV)
            if guard_owner and guard_owner != str(os.getpid()):
                raise RuntimeError(_MAIN_GUARD_HINT)
            os.environ[_STRATEGY_GUARD_ENV] = str(os.getpid())
            try:
                # Python 3.12 warns when forking from a multi-threaded process,
                # so spawn explicitly rather than inheriting the platform default.
                with ProcessPoolExecutor(cores, mp_context=get_context("spawn")) as own_executor:
                    self._run_stages(tasks, own_executor, cores, verbose)
            finally:
                os.environ.pop(_STRATEGY_GUARD_ENV, None)
        else:
            self._run_serially(tasks, verbose)

        self._df.attrs["_ta_last_run"] = get_time(self.exchange, to_string=True)

        if verbose:
            logger.info(f"Total indicators: {len(ta)}")
            logger.info(f"Columns added: {len(self._df.columns) - initial_column_count}")
            logger.info(f"Last Run: {self.last_run}")
        if timed:
            logger.info(f"Runtime: {final_time(stime)}")

        if returns:
            return self._df

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for private/dunder attributes
        if name.startswith("_"):
            raise AttributeError(name)
        func = _find_indicator_func(name)
        if func is None:
            # A property getter raising AttributeError lands here too: Python
            # cannot tell "no such attribute" from "the descriptor failed", so
            # reporting a missing attribute would replace the real error (and
            # its traceback) with a false one.  Re-run the descriptor directly
            # -- that path bypasses __getattr__ -- to surface the actual cause.
            descriptor = getattr(type(self), name, None)
            if descriptor is not None:
                return descriptor.__get__(self, type(self))
            raise AttributeError(f"'AnalysisIndicators' object has no attribute '{name}'")
        wrapper = _make_ta_wrapper(func)
        wrapper.__name__ = name
        wrapper.__qualname__ = f"AnalysisIndicators.{name}"
        # Cache on the class so future calls bypass __getattr__.
        # Aliases (max/min/sum) are intentionally excluded: caching them on
        # the class would permanently shadow Python builtins at the class level.
        if name not in _MATH_ALIASES:
            setattr(type(self), name, wrapper)
        return wrapper.__get__(self, type(self))

    # ichimoku is the only explicit wrapper left: the underlying function still
    # supports a deprecated (visible, span) tuple return. This wrapper pins the
    # single-DataFrame return (as_dataframe=True) and forwards append_span, so
    # _post_process can handle it like any other indicator.
    def ichimoku(
        self,
        tenkan=None,
        kijun=None,
        senkou=None,
        include_chikou=True,
        append_span: bool = False,
        offset=None,
        **kwargs,
    ):
        """Ichimoku Kinkō Hyō.

        Returns a single DataFrame of the visible period columns. Pass
        append_span=True to also append the future-dated span rows (projected
        Senkou A/B for the next kijun periods).
        """
        from pandas_ta_classic.overlap.ichimoku import ichimoku as _ichimoku

        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", self._default_column("close")))
        result = _ichimoku(
            high=high,
            low=low,
            close=close,
            tenkan=tenkan,
            kijun=kijun,
            senkou=senkou,
            include_chikou=include_chikou,
            offset=offset,
            as_dataframe=True,
            append_span=append_span,
            **kwargs,
        )
        self._add_prefix_suffix(result, **kwargs)
        self._append(result, **kwargs)
        return self._post_process(result, **kwargs)
