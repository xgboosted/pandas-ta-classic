import inspect
import logging
from collections.abc import Hashable
from copy import copy
from dataclasses import dataclass, field
from multiprocessing import cpu_count, get_context
from numbers import Integral
from time import perf_counter
from typing import Any
from warnings import simplefilter

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ta_classic._indicator_loader import (
    _COLUMN_KWARG_KEYS,
    _COLUMN_PARAM_TO_COL_KEY,
    _DEFAULT_COLUMN_NAMES,
    _find_indicator_func,
    _make_ta_wrapper,
)
from pandas_ta_classic._meta import _MATH_ALIASES, EXCHANGE_TZ, Category, Imports, version
from pandas_ta_classic.utils import final_time, get_time, is_datetime_ordered, to_utc, total_time
from pandas_ta_classic.utils._core import _bool_param, _pos_int
from pandas_ta_classic.utils._time import TIME_RANGE_UNITS

logger = logging.getLogger(__name__)


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
        if len(kwargs["col_names"]) != len(result.columns):
            # too few used to be logged and nothing appended; extras were ignored
            raise ValueError(f"col_names has {len(kwargs['col_names'])} name(s) for {len(result.columns)} column(s): {list(result.columns)}")
        for col, ind_name in zip(result.columns, kwargs["col_names"]):
            df[ind_name] = result.loc[:, col]
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
    >>> ichimoku = ta.ichimoku(df["High"], df["Low"], df["Close"])

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
    _cores = cpu_count()
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
        timed = _bool_param(timed, False, "timed")
        if "version" in kwargs:
            # the alias was removed in 0.9.0; the indicator's **kwargs would swallow it
            raise TypeError("df.ta() no longer accepts 'version': use show_version=True")
        if show_version:
            logger.info(f"Pandas TA - Technical Analysis Indicators - v{self.version}")
        if kind is None:
            # "General Help": list the indicators (this called a missing help() method)
            return self.indicators()
        if not isinstance(kind, str):
            raise TypeError(f"df.ta() kind must be an indicator name, got {kind!r}")
        kind = kind.lower()
        fn = getattr(self, kind, None)
        if fn is None or not callable(fn):
            # an unknown name used to log an error and return None
            raise ValueError(f"df.ta() kind={kind!r} is not an indicator; see df.ta.indicators()")

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
        """property: df.ta.cores = integer (0 disables multiprocessing; capped at cpu_count(); None resets)"""
        cpus = cpu_count()
        if value is None:
            self._df.attrs["_ta_cores"] = cpus
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
        """Sets the DataFrame's index to UTC (localises a naive index, converts an aware one).

        This changes ``df`` itself, as documented. It had stopped doing so when
        ``ta.to_utc()`` began returning a copy: it rebound the accessor's own
        reference instead, which pandas 3 discards immediately.
        """
        self._df.index = to_utc(self._df).index

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
        # pandas copies attrs into df.copy() and slices; storing this frame's
        # id keeps chain mode on the frame that asked for it.
        # ponytail: id() can be reused after the chained frame is freed; key by a weakref if that ever matters (attrs must stay picklable for strategy()).
        self._df.attrs["_ta_chain"] = id(self._df)
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
        if not _bool_param(kwargs.get("append"), False, "append"):
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
            if "col_names" in kwargs and len(kwargs["col_names"]) != 1:
                raise ValueError(f"col_names has {len(kwargs['col_names'])} names for one column ({result.name})")
            ind_name = kwargs["col_names"][0] if "col_names" in kwargs else result.name
            df[ind_name] = result

    def _default_column(self, name: str) -> str:
        """The column an indicator reads when the caller names none.

        'close' resolves to df.ta.adjusted when that is set; every other
        column name is returned unchanged.
        """
        if name == "close" and self.adjusted is not None:
            # A copy or slice inherits _ta_adjusted from df.attrs but may lack
            # the adjusted column. Computing on 'close' instead would silently
            # swap adjusted prices for unadjusted ones.
            if self.adjusted not in self._df.columns:
                raise KeyError(f"df.ta.adjusted is {self.adjusted!r}, but the DataFrame has no such column; set df.ta.adjusted = None to use 'close'")
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
        # None means "not provided" (an optional input). It used to return the
        # adjusted column, so open=None could read adjusted close as open.
        if series is None:
            return None
        # Ok.  So it's a str.
        if isinstance(series, str):
            # Return the df column since it's in there.
            if series in df.columns:
                return df[series]
            # Attempt to match the 'series' because it was likely misspelled:
            # case-insensitive exact match only.  A prefix match (str.match)
            # resolved 'open' to a leading 'Open time' column on Binance-style
            # frames, silently returning the epoch-milliseconds column.
            matches = [col for col in df.columns if isinstance(col, str) and col.lower() == series.lower()]
            if matches:
                return df[matches[0]]
            cols = ", ".join(str(c) for c in df.columns)
            raise KeyError(f"Column {series!r} not found. Available columns: {cols}")
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
        # Case-insensitive exact match, mirroring _get_column()'s fallback.
        matches = [col for col in columns if isinstance(col, str) and col.lower() == name.lower()]
        return matches[0] if matches else None

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

    def _mp_worker(self, arguments: tuple):
        """Multiprocessing Worker to handle different Methods."""
        method, args, kwargs = arguments
        return getattr(self, method)(*args, **kwargs)

    def _post_process(self, result, **kwargs) -> pd.Series | pd.DataFrame | None:
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        * In chain mode, auto-appends and returns the DataFrame for fluent chaining.
        """
        verbose = _bool_param(kwargs.pop("verbose", None), False, "verbose")
        chain_mode = self._df.attrs.get("_ta_chain") == id(self._df)

        if not isinstance(result, (pd.Series, pd.DataFrame)):
            if verbose:
                logger.error("The result was not a Series or DataFrame.")
            # An indicator that returns None (missing input, short series) must
            # not be silently replaced by the caller's whole DataFrame.
            return None
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
            if arg.lower() in self.categories:
                name, mode["category"] = arg, True
        if isinstance(arg, Strategy):
            strategy_ = arg
            if strategy_.ta is None or strategy_.name.lower() == "all":
                mode["all"] = True
            elif strategy_.name.lower() in self.categories:
                name, mode["category"] = strategy_.name, True
            else:
                name, mode["custom"] = strategy_.name, True
        if not any(mode.values()):
            # an unknown name used to log an error and return None
            raise ValueError(f"strategy() needs 'all', a category ({', '.join(self.categories)}) or a Strategy, got {arg!r}")
        return name, mode

    def _validate_exclude(self, exclude, caller: str) -> list:
        """Return *exclude* as a list of known indicator names; raise for anything else."""
        if not isinstance(exclude, (list, tuple, set)) or not all(isinstance(x, str) for x in exclude):
            raise TypeError(f"{caller}() exclude must be a list of indicator names, got {exclude!r}")
        unknown = sorted(set(exclude) - {i for names in Category.values() for i in names})
        if unknown:
            raise ValueError(f"{caller}() exclude has unknown indicator name(s): {unknown}")
        return list(exclude)

    def _missing_required_column(self, name: str, kwargs: dict) -> bool:
        """True when indicator *name* needs a column (no default) that the DataFrame does not have."""
        func = _find_indicator_func(name)
        if func is None:
            raise ValueError(f"unknown indicator {name!r}")
        sig = inspect.signature(func)
        for param, spec in sig.parameters.items():
            col_key = _COLUMN_PARAM_TO_COL_KEY.get(param)
            if col_key is None or spec.default is not inspect.Parameter.empty:
                continue
            column = kwargs.get(col_key, self._default_column(col_key))
            if isinstance(column, str) and self._matching_column(column) is None:
                return True
        return False

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
        as_list = _bool_param(kwargs.get("as_list"), False, "as_list")
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
        # a tuple or a misspelled name used to be ignored silently
        removed += self._validate_exclude(kwargs.get("exclude", []), "indicators")

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


        Kwargs:
            chunksize (bool): Adjust the chunksize for the Multiprocessing Pool.
                Default: Number of cores of the OS
            exclude (list): List of indicator names to exclude. Some are
                excluded by default for various reasons; they require additional
                sources, performance (td_seq), not a ohlcv chart (vp) etc.
            name (str): Select all indicators or indicators by
                Category such as: "candles", "cycles", "momentum", "overlap",
                "performance", "statistics", "trend", "volatility", "volume", or
                "all". Default: "all"
            ordered (bool): Whether to run "all" in order. Default: True
            timed (bool): Show the process time of the strategy().
                Default: False
            verbose (bool): Provide some additional insight on the progress of
                the strategy() execution. Default: False
        """
        # If True, it returns the resultant DataFrame. Default: False
        returns = _bool_param(kwargs.pop("returns", None), False, "returns")
        # Ensure indicators are appended to the DataFrame
        kwargs["append"] = True
        all_ordered = _bool_param(kwargs.pop("ordered", None), True, "ordered")
        mp_chunksize = _pos_int(kwargs.pop("chunksize", None), max(self.cores, 1), "chunksize")

        # Initialize
        initial_column_count = len(self._df.columns)
        excluded = [
            "long_run",
            "mavp",  # Requires a per-bar 'periods' Series
            "short_run",
            "td_seq",  # Performance exclusion
            "tsignals",
            "vp",
            "xsignals",
        ]

        # Get the Strategy Name and mode
        name, mode = self._strategy_mode(*args)

        # If All or a Category, exclude user list if any
        if not isinstance(self._df.index, pd.DatetimeIndex):
            excluded.append("vwap")  # anchors by calendar period; raises without a DatetimeIndex
        user_excluded = self._validate_exclude(kwargs.pop("exclude", []), "strategy")
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
        else:
            ta = self.indicators(as_list=True, exclude=excluded)

        if mode["all"] or mode["category"]:
            # Skip, and report, indicators needing a column this DataFrame does
            # not have (volume on OHLC-only data): each would raise KeyError.
            unavailable = [x for x in ta if self._missing_required_column(x, kwargs)]
            ta = [x for x in ta if x not in unavailable]
            excluded += unavailable

        verbose = _bool_param(kwargs.pop("verbose", None), False, "verbose")
        if verbose:
            logger.info(f"Strategy: {name}\nIndicator arguments: {kwargs}")
            if mode["all"] or mode["category"]:
                excluded_str = ", ".join(excluded)
                logger.info(f"Excluded[{len(excluded)}]: {excluded_str}")

        timed = _bool_param(kwargs.pop("timed", None), False, "timed")
        results: Any = []
        use_multiprocessing = self.cores > 0
        has_col_names = False

        if timed:
            stime = perf_counter()

        if use_multiprocessing and mode["custom"]:
            # Determine if the Custom Model has 'col_names' parameter
            has_col_names = bool(len([True for x in ta if "col_names" in x and isinstance(x["col_names"], tuple)]))
            # A chained entry reads a column an earlier entry appends (close="CUMLOGRET_1").
            # Each Pool worker holds its own copy of the input, so the chain only works serially.
            chained = any(isinstance(ind.get(key), str) and self._matching_column(ind[key]) is None for ind in ta for key in _COLUMN_KWARG_KEYS)

            if has_col_names or chained:
                use_multiprocessing = False

        if Imports["tqdm"]:
            from tqdm import tqdm  # type: ignore[import-untyped]  # optional; ships no stubs

        if use_multiprocessing:
            _total_ta = len(ta)

            # Create a lightweight copy of self holding only the columns an
            # indicator can reach (see _worker_columns).  Without this, each
            # imap() call pickles the whole of self._df -- which grows as
            # indicators are appended -- causing pandas BlockManager integrity
            # errors in workers and pool deadlocks.  On Windows the oversized
            # pickle also fails the overlapped WriteFile on the task pipe
            # outright with 'OSError: [WinError 1450] Insufficient system
            # resources'.
            kwarg_sources = [kwargs, *ta] if mode["custom"] else [kwargs]
            slim = copy(self)
            slim._df = self._df[self._worker_columns(kwarg_sources)].copy()

            # Python 3.12 warns when forking from a multi-threaded process.
            # Use spawn context explicitly to avoid unsafe fork behavior.
            pool = get_context("spawn").Pool(self.cores)
            try:
                # Some magic to optimize chunksize for speed based on total ta indicators
                _chunksize = mp_chunksize - 1 if mp_chunksize > _total_ta else int(np.log10(_total_ta)) + 1
                if verbose:
                    logger.info(f"Multiprocessing {_total_ta} indicators with {_chunksize} chunks and {self.cores}/{cpu_count()} cpus.")

                results = None
                if mode["custom"]:
                    # Create a list of all the custom indicators into a list
                    custom_ta = [
                        (
                            ind["kind"],
                            _strategy_params(ind),
                            {**ind, **kwargs},
                        )
                        for ind in ta
                    ]
                    # Custom multiprocessing pool. Must be ordered for Chained Strategies
                    results = pool.imap(slim._mp_worker, custom_ta, _chunksize)
                else:
                    default_ta: list = [(ind, (), kwargs) for ind in ta]
                    # All and Categorical multiprocessing pool.
                    if all_ordered:
                        if Imports["tqdm"] and verbose:
                            results = tqdm(pool.imap(slim._mp_worker, default_ta, _chunksize))  # Order over Speed
                        else:
                            results = pool.imap(slim._mp_worker, default_ta, _chunksize)  # Order over Speed
                    else:
                        if Imports["tqdm"] and verbose:
                            results = tqdm(pool.imap_unordered(slim._mp_worker, default_ta, _chunksize))  # Speed over Order
                        else:
                            results = pool.imap_unordered(slim._mp_worker, default_ta, _chunksize)  # Speed over Order
                if results is None:
                    logger.warning(f"ta.strategy('{name}') has no results.")
                    pool.terminate()
                    return

                # Consume the lazy iterator while the pool is still alive.
                [self._post_process(r, **kwargs) for r in results]
                pool.close()
            except Exception:
                pool.terminate()
                raise
            finally:
                pool.join()

            del slim
            self._df.attrs["_ta_last_run"] = get_time(self.exchange, to_string=True)

        else:
            # Without multiprocessing:
            if verbose:
                _col_msg = "[i] No mulitproccessing (cores = 0)."
                if has_col_names:
                    _col_msg = "[i] No mulitproccessing support for 'col_names' option."
                logger.info(_col_msg)

            if mode["custom"]:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, "[i] Progress")
                    for ind in pbar:
                        params = _strategy_params(ind)
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
                else:
                    for ind in ta:
                        params = _strategy_params(ind)
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
            else:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, "[i] Progress")
                    for ind in pbar:
                        getattr(self, ind)(*(), **kwargs)
                else:
                    for ind in ta:
                        getattr(self, ind)(*(), **kwargs)
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
