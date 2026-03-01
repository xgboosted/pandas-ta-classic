# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass, field
from multiprocessing import cpu_count, Pool
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Tuple
from warnings import catch_warnings, simplefilter

logger = logging.getLogger(__name__)

import pandas as pd
from numpy import log10 as npLog10
from numpy import ndarray as npNdarray
from pandas_ta_classic._meta import Category, EXCHANGE_TZ, Imports, version
from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS
from pandas_ta_classic.candles import *
from pandas_ta_classic.cycles import *
from pandas_ta_classic.momentum import *
from pandas_ta_classic.overlap import *
from pandas_ta_classic.performance import *
from pandas_ta_classic.statistics import *
from pandas_ta_classic.trend import *
from pandas_ta_classic.volatility import *
from pandas_ta_classic.volume import *

# TODO: These wildcard imports cause name collisions between indicator category
# sub-packages and same-named functions from utils._metrics / utils._signals.
# Affected names: `volatility` (shadows pandas_ta_classic.volatility subpackage),
# `signals` (shadows pandas_ta_classic.utils._signals.signals).
# Fix requires replacing wildcard imports with explicit per-name imports (2.2).
from pandas_ta_classic.utils import *

df = pd.DataFrame()


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
    ta: List = field(default_factory=list)  # Required.
    # Helpful. More descriptive version or notes or w/e.
    description: str = "TA Description"
    # Optional. Gets Exchange Time and Local Time execution time
    created: Optional[str] = get_time(to_string=True)

    def __post_init__(self):
        required_args = ["[X] Strategy requires the following argument(s):"]

        name_is_str = isinstance(self.name, str)
        ta_is_list = isinstance(self.ta, list)

        if self.name is None or not name_is_str:
            required_args.append(
                ' - name. Must be a string. Example: "My TA". Note: "all" is reserved.'
            )

        if self.ta is None:
            self.ta = None
        elif self.ta is not None and ta_is_list and self.total_ta() > 0:
            pass  # Valid ta list; element-level validation left to indicator calls
        else:
            s = " - ta. Format is a list of dicts. Example: [{'kind': 'sma', 'length': 10}]"
            s += "\n       Check the indicator for the correct arguments if you receive this error."
            required_args.append(s)

        if len(required_args) > 1:
            for _msg in required_args:
                logger.error(_msg)
            return None

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


import inspect as _inspect

# ─────────────────────────────────────────────────────────────────────────────
# Auto-wrapper infrastructure
# Maps indicator function parameter names that represent OHLCV Series to the
# corresponding DataFrame column name used in kwargs.pop(col_name, col_name).
# ─────────────────────────────────────────────────────────────────────────────
_SERIES_PARAM_MAP = {
    "open_": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def _make_indicator_method(func):
    """Create an AnalysisIndicators method that auto-wraps an indicator function.

    Inspects *func*'s signature to determine which parameters are OHLCV Series
    (i.e. named ``open_``, ``high``, ``low``, ``close``, or ``volume`` **and**
    are required — no ``None`` default).  Those series are extracted from the
    DataFrame via ``_get_column``; all remaining kwargs are forwarded to the
    underlying function unchanged.

    If the function does not accept ``**kwargs`` (e.g. ``pvr``), only the
    explicitly declared parameters are forwarded so that meta-kwargs like
    ``append`` or ``prefix`` do not cause a ``TypeError``.

    The generated method is cached on the class after first use so that
    subsequent calls bypass ``__getattr__``.
    """
    sig = _inspect.signature(func)
    params_list = list(sig.parameters.items())
    # Collect ALL (param_name, df_col_name) pairs for OHLCV series params,
    # including those with default=None.  Rationale: some indicators declare
    # optional OHLCV params (e.g. rvi.high/low) that the accessor must still
    # always pull from the DataFrame to match the behaviour of the original
    # hand-written wrappers.  Indicators where an optional series must NOT be
    # extracted by default (e.g. psar.close) are handled by explicit wrappers
    # in the class body and never reach this code path.
    series_params = [
        (pname, _SERIES_PARAM_MAP[pname])
        for pname, param in params_list
        if pname in _SERIES_PARAM_MAP
    ]
    # Non-series positional params, in declaration order.  Used to map
    # Strategy ``params`` tuples (positional args) to keyword args so that the
    # generated method works with both ``df.ta.ema(length=5)`` and the
    # multiprocessing worker calling ``self.ema(5, ...)``.
    non_series_positional = [
        pname
        for pname, param in params_list
        if pname not in _SERIES_PARAM_MAP
        and param.kind
        in (
            _inspect.Parameter.POSITIONAL_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    # Does the function accept arbitrary keyword arguments?
    _has_var_kw = any(
        p.kind is _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    # For functions without **kwargs, remember which named params they accept.
    _known_params = None if _has_var_kw else frozenset(sig.parameters.keys())

    def method(self, *args, **kwargs):
        # Map positional args to non-series param names (Strategy 'params' support).
        for pname, val in zip(non_series_positional, args):
            kwargs.setdefault(pname, val)
        call_kwargs = {}
        for param_name, col_name in series_params:
            col_key = kwargs.pop(col_name, col_name)
            call_kwargs[param_name] = self._get_column(col_key)
        if _has_var_kw:
            result = func(**call_kwargs, **kwargs)
        else:
            # Filter to only the params the function declares (drops append, etc.)
            filtered = {k: v for k, v in kwargs.items() if k in _known_params}
            result = func(**call_kwargs, **filtered)
        return self._post_process(result, **kwargs)

    method.__name__ = func.__name__
    method.__qualname__ = f"AnalysisIndicators.{func.__name__}"
    method.__doc__ = func.__doc__
    return method


# Pandas TA - DataFrame Analysis Indicators
@pd.api.extensions.register_dataframe_accessor("ta")
class AnalysisIndicators:
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
    >>> ichimoku, span = ta.ichimoku(df["High"], df["Low"], df["Close"])

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
        KC, et al will return a Pandas DataFrame. Ichimoku on the other hand
        will return two DataFrames, the Ichimoku DataFrame for the known period
        and a Span DataFrame for the future of the Span values.

    Let's get started!

    1. Loading the 'ta' module:
    >>> import pandas as pd
    >>> import pandas_ta_classic

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
    _pending_appends: Optional[List[pd.DataFrame]] = None
    _time_range = "years"
    _last_run = get_time(_exchange, to_string=True)

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._df = pandas_obj
        self._last_run = get_time(self._exchange, to_string=True)

    @staticmethod
    def _validate(obj: Tuple[pd.DataFrame, pd.Series]):
        if not isinstance(obj, pd.DataFrame) and not isinstance(obj, pd.Series):
            raise AttributeError("[X] Must be either a Pandas Series or DataFrame.")

    # DataFrame Behavioral Methods
    def __call__(
        self,
        kind: Optional[str] = None,
        timed: bool = False,
        version: bool = False,
        **kwargs,
    ):
        if version:
            logger.info("Pandas TA - Technical Analysis Indicators - v%s", self.version)
        try:
            if isinstance(kind, str):
                kind = kind.lower()
                fn = getattr(self, kind)

                if timed:
                    stime = perf_counter()

                # Run the indicator
                result = fn(**kwargs)  # = getattr(self, kind)(**kwargs)
                self._last_run = get_time(
                    self.exchange, to_string=True
                )  # Save when it completed it's run

                if timed:
                    result.timed = final_time(stime)
                    logger.debug("[+] %s: %s", kind, result.timed)

                return result
            else:
                self.help()

        except Exception:
            logger.exception("Error running indicator '%s'", kind)

    # Public Get/Set DataFrame Properties
    @property
    def adjusted(self) -> Optional[str]:
        """property: df.ta.adjusted"""
        return self._adjusted

    @adjusted.setter
    def adjusted(self, value: str) -> None:
        """property: df.ta.adjusted = 'adj_close'"""
        if value is not None and isinstance(value, str):
            self._adjusted = value
        else:
            self._adjusted = None

    @property
    def cores(self) -> int:
        """Returns the categories."""
        return self._cores

    @cores.setter
    def cores(self, value: int) -> None:
        """property: df.ta.cores = integer"""
        cpus = cpu_count()
        if value is not None and isinstance(value, int):
            self._cores = int(value) if 0 <= value <= cpus else cpus
        else:
            self._cores = cpus

    @property
    def exchange(self) -> str:
        """Returns the current Exchange. Default: "NYSE"."""
        return self._exchange

    @exchange.setter
    def exchange(self, value: str) -> None:
        """property: df.ta.exchange = "LSE" """
        if value is not None and isinstance(value, str) and value in EXCHANGE_TZ.keys():
            self._exchange = value

    @property
    def last_run(self) -> Optional[str]:
        """Returns the time when the DataFrame was last run."""
        return self._last_run

    # Public Get DataFrame Properties
    @property
    def categories(self) -> List[str]:
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
        return total_time(self._df, self._time_range)

    @time_range.setter
    def time_range(self, value: str) -> None:
        """property: df.ta.time_range = "years" (Default)"""
        if value is not None and isinstance(value, str):
            self._time_range = value
        else:
            self._time_range = "years"

    @property
    def to_utc(self) -> None:
        """Sets the DataFrame index to UTC format"""
        self._df = to_utc(self._df)

    @property
    def version(self) -> str:
        """Returns the version."""
        return version

    # Private DataFrame Methods
    def _add_prefix_suffix(self, result=None, **kwargs) -> None:
        """Add prefix and/or suffix to the result columns"""
        if result is None:
            return
        else:
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

    @staticmethod
    def _build_append_fragment(result, **kwargs):
        """Build a DataFrame fragment from an indicator result, applying col_names.

        Returns a DataFrame ready for concat, or None if inputs are invalid.
        """
        if "col_names" in kwargs and not isinstance(kwargs["col_names"], tuple):
            kwargs["col_names"] = (kwargs["col_names"],)

        if isinstance(result, pd.DataFrame):
            if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple):
                if len(kwargs["col_names"]) >= len(result.columns):
                    renamed = result.copy()
                    renamed.columns = list(kwargs["col_names"][: len(result.columns)])
                    return renamed
                else:
                    logger.warning(
                        "Not enough col_names were specified: got %d, expected %d.",
                        len(kwargs["col_names"]),
                        len(result.columns),
                    )
                    return None
            else:
                return result
        else:
            ind_name = (
                kwargs["col_names"][0]
                if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple)
                else result.name
            )
            return result.rename(ind_name).to_frame()

    def _append(self, result=None, **kwargs) -> None:
        """Appends a Pandas Series or DataFrame columns to self._df."""
        if "append" in kwargs and kwargs["append"]:
            df = self._df
            if df is None or result is None:
                return
            else:
                fragment = self._build_append_fragment(result, **kwargs)
                if fragment is None:
                    return

                if self._pending_appends is not None:
                    self._pending_appends.append(fragment)
                else:
                    with catch_warnings():
                        simplefilter(
                            action="ignore", category=pd.errors.PerformanceWarning
                        )
                        for col in fragment.columns:
                            df[col] = fragment[col]

    def _check_na_columns(self, stdout: bool = True):
        """Returns the columns in which all it's values are na."""
        return [x for x in self._df.columns if all(self._df[x].isna())]

    def _get_column(self, series):
        """Attempts to get the correct series or 'column' and return it."""
        df = self._df
        if df is None:
            return

        # Explicitly passing a pd.Series to override default.
        if isinstance(series, pd.Series):
            return series
        # Apply default if no series nor a default.
        elif series is None:
            return df[self.adjusted] if self.adjusted is not None else None
        # Ok.  So it's a str.
        elif isinstance(series, str):
            # Return the df column since it's in there.
            if series in df.columns:
                return df[series]
            else:
                # Attempt to match the 'series' because it was likely
                # misspelled.
                matches = df.columns.str.match(series, case=False)
                match = [i for i, x in enumerate(matches) if x]
                # If found, awesome.  Return it or return the 'series'.
                cols = ", ".join(list(df.columns))
                NOT_FOUND = f"[X] Ooops!!! It's {series not in df.columns}, the series '{series}' was not found in {cols}"
                if len(match):
                    return df.iloc[:, match[0]]
                logger.warning(NOT_FOUND)
                return None

    def _indicators_by_category(self, name: str) -> Optional[list]:
        """Returns indicators by Categorical name."""
        return Category[name] if name in self.categories else None

    def _mp_worker(self, arguments: tuple):
        """Multiprocessing Worker to handle different Methods."""
        method, args, kwargs = arguments

        if method != "ichimoku":
            return getattr(self, method)(*args, **kwargs)
        else:
            return getattr(self, method)(*args, **kwargs)[0]

    def _post_process(self, result, **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        """
        verbose = kwargs.pop("verbose", False)
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            if verbose:
                logger.debug("[X] Oops! The result was not a Series or DataFrame.")
            return self._df
        else:
            # Append only specific columns to the dataframe (via
            # 'col_numbers':(0,1,3) for example)
            result = (
                result.iloc[:, [int(n) for n in kwargs["col_numbers"]]]
                if isinstance(result, pd.DataFrame)
                and "col_numbers" in kwargs
                and kwargs["col_numbers"] is not None
                else result
            )
            # Add prefix/suffix and append to the dataframe
            self._add_prefix_suffix(result=result, **kwargs)
            self._append(result=result, **kwargs)
        return result

    def _strategy_mode(self, *args) -> tuple:
        """Helper method to determine the mode and name of the strategy. Returns tuple: (name:str, mode:dict)"""
        name = "All"
        mode = {"all": False, "category": False, "custom": False}

        if len(args) == 0:
            mode["all"] = True
        else:
            if isinstance(args[0], str):
                if args[0].lower() == "all":
                    name, mode["all"] = name, True
                if args[0].lower() in self.categories:
                    name, mode["category"] = args[0], True

            if isinstance(args[0], Strategy):
                strategy_ = args[0]
                if strategy_.ta is None or strategy_.name.lower() == "all":
                    name, mode["all"] = name, True
                elif strategy_.name.lower() in self.categories:
                    name, mode["category"] = strategy_.name, True
                else:
                    name, mode["custom"] = strategy_.name, True

        return name, mode

    # Public DataFrame Methods
    def constants(self, append: bool, values: list):
        """Constants

        Add or remove constants to the DataFrame easily with Numpy's arrays or
        lists. Useful when you need easily accessible horizontal lines for
        charting.

        Add constant '1' to the DataFrame
        >>> df.ta.constants(True, [1])
        Remove constant '1' to the DataFrame
        >>> df.ta.constants(False, [1])

        Adding constants for charting
        >>> import numpy as np
        >>> chart_lines = np.append(np.arange(-4, 5, 1), np.arange(-100, 110, 10))
        >>> df.ta.constants(True, chart_lines)
        Removing some constants from the DataFrame
        >>> df.ta.constants(False, np.array([-60, -40, 40, 60]))

        Args:
            append (bool): If True, appends a Numpy range of constants to the
                working DataFrame.  If False, it removes the constant range from
                the working DataFrame. Default: None.

        Returns:
            Returns the appended constants
            Returns nothing to the user.  Either adds or removes constant ranges
            from the working DataFrame.
        """
        if isinstance(values, npNdarray) or isinstance(values, list):
            if append:
                for x in values:
                    self._df[f"{x}"] = x
                return self._df[self._df.columns[-len(values) :]]
            else:
                for x in values:
                    del self._df[f"{x}"]

    def __dir__(self):
        """Include all Category indicator names so that dir(df.ta) is complete."""
        base = list(super().__dir__())
        for cat_indicators in Category.values():
            for name in cat_indicators:
                if name not in base:
                    base.append(name)
        return sorted(base)

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

        # Build indicator list from Category (auto-discovered, always up to date).
        ta_indicators = sorted(name for cat in Category.values() for name in cat)

        # Apply user exclusions.
        user_excluded = kwargs.setdefault("exclude", [])
        if isinstance(user_excluded, list) and len(user_excluded) > 0:
            ta_indicators = [x for x in ta_indicators if x not in user_excluded]

        # If as a list, immediately return
        if as_list:
            return ta_indicators

        total_indicators = len(ta_indicators)
        header = f"Pandas TA - Technical Analysis Indicators - v{self.version}"
        s = f"{header}\nTotal Indicators & Utilities: {total_indicators + len(ALL_PATTERNS)}\n"
        if total_indicators > 0:
            print(
                f"{s}Abbreviations:\n    {', '.join(ta_indicators)}"
                f"\n\nCandle Patterns:\n    {', '.join(ALL_PATTERNS)}"
            )
        else:
            print(s)  # intentional: indicators() is an explicit user-facing listing

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
        returns = kwargs.pop("returns", False)
        # cpus = cpu_count()
        # Ensure indicators are appended to the DataFrame
        kwargs["append"] = True
        all_ordered = kwargs.pop("ordered", True)
        mp_chunksize = kwargs.pop("chunksize", self.cores)

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
            "short_run",
            "td_seq",  # Performance exclusion
            "tsignals",
            "vp",
            "xsignals",
        ]

        # Get the Strategy Name and mode
        name, mode = self._strategy_mode(*args)

        # If All or a Category, exclude user list if any
        user_excluded = kwargs.pop("exclude", [])
        if mode["all"] or mode["category"]:
            excluded += user_excluded

        # Collect the indicators, remove excluded or include kwarg["append"]
        if mode["category"]:
            ta = self._indicators_by_category(name.lower())
            for x in excluded:
                if x in ta:
                    ta.remove(x)
        elif mode["custom"]:
            ta = args[0].ta
            for kwds in ta:
                kwds["append"] = True
        elif mode["all"]:
            ta = self.indicators(as_list=True, exclude=excluded)
        else:
            logger.error("[X] Not an available strategy.")
            return None

        # Remove Custom indicators with "length" keyword when larger than the DataFrame
        # Possible to have other indicator main window lengths to be included
        removal = []
        for kwds in ta:
            _ = False
            if "length" in kwds and kwds["length"] > self._df.shape[0]:
                _ = True
            if _:
                removal.append(kwds)
        if len(removal) > 0:
            for x in removal:
                ta.remove(x)

        verbose = kwargs.pop("verbose", False)
        if verbose:
            logger.info("[+] Strategy: %s\n[i] Indicator arguments: %s", name, kwargs)
            if mode["all"] or mode["category"]:
                excluded_str = ", ".join(excluded)
                logger.info("[i] Excluded[%d]: %s", len(excluded), excluded_str)

        timed = kwargs.pop("timed", False)
        results: Any = []
        use_multiprocessing = True if self.cores > 0 else False
        has_col_names = False

        if timed:
            stime = perf_counter()

        if use_multiprocessing and mode["custom"]:
            # Determine if the Custom Model has 'col_names' parameter
            has_col_names = (
                True
                if len(
                    [
                        True
                        for x in ta
                        if "col_names" in x and isinstance(x["col_names"], tuple)
                    ]
                )
                else False
            )

            if has_col_names:
                use_multiprocessing = False

        if Imports["tqdm"]:
            # from tqdm import tqdm
            from tqdm import tqdm

        # Enable deferred batching for all/category modes.
        # Custom mode stays immediate so chained indicators can reference
        # columns produced by earlier ones.
        if not mode["custom"]:
            self._pending_appends: List[pd.DataFrame] = []

        if use_multiprocessing:
            _total_ta = len(ta)

            # Create a lightweight copy of self that contains only the
            # original OHLCV columns.  Without this, each imap() call
            # pickles self._df (which grows as indicators are appended),
            # causing pandas BlockManager integrity errors in workers and
            # pool deadlocks.
            slim = self.__class__.__new__(self.__class__)
            slim.__dict__.update(self.__dict__)
            slim._df = self._df[self._df.columns[:initial_column_count]].copy()

            pool = Pool(self.cores)
            try:
                # Some magic to optimize chunksize for speed based on total ta indicators
                _chunksize = (
                    mp_chunksize - 1
                    if mp_chunksize > _total_ta
                    else int(npLog10(_total_ta)) + 1
                )
                if verbose:
                    logger.info(
                        "[i] Multiprocessing %d indicators with %d chunks and %d/%d cpus.",
                        _total_ta,
                        _chunksize,
                        self.cores,
                        cpu_count(),
                    )

                results = None
                if mode["custom"]:
                    # Create a list of all the custom indicators into a list
                    custom_ta = [
                        (
                            ind["kind"],
                            (
                                ind["params"]
                                if "params" in ind and isinstance(ind["params"], tuple)
                                else ()
                            ),
                            {**ind, **kwargs},
                        )
                        for ind in ta
                    ]
                    # Custom multiprocessing pool. Must be ordered for Chained Strategies
                    results = pool.imap(slim._mp_worker, custom_ta, _chunksize)
                else:
                    default_ta: list = [(ind, tuple(), kwargs) for ind in ta]
                    # All and Categorical multiprocessing pool.
                    if all_ordered:
                        if Imports["tqdm"]:
                            results = tqdm(
                                pool.imap(slim._mp_worker, default_ta, _chunksize)
                            )  # Order over Speed
                        else:
                            results = pool.imap(
                                slim._mp_worker, default_ta, _chunksize
                            )  # Order over Speed
                    else:
                        if Imports["tqdm"]:
                            results = tqdm(
                                pool.imap_unordered(
                                    slim._mp_worker, default_ta, _chunksize
                                )
                            )  # Speed over Order
                        else:
                            results = pool.imap_unordered(
                                slim._mp_worker, default_ta, _chunksize
                            )  # Speed over Order
                if results is None:
                    logger.error("[X] ta.strategy('%s') has no results.", name)
                    self._pending_appends = None
                    return

                # Consume the lazy iterator while the pool is still alive.
                [self._post_process(r, **kwargs) for r in results]
            finally:
                pool.terminate()
                pool.join()

            del slim
            self._last_run = get_time(self.exchange, to_string=True)

        else:
            # Without multiprocessing:
            if verbose:
                if has_col_names:
                    logger.info(
                        "[i] No multiprocessing support for 'col_names' option."
                    )
                else:
                    logger.info("[i] No multiprocessing (cores = 0).")

            if mode["custom"]:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, f"[i] Progress")
                    for ind in pbar:
                        params = (
                            ind["params"]
                            if "params" in ind and isinstance(ind["params"], tuple)
                            else tuple()
                        )
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
                else:
                    for ind in ta:
                        params = (
                            ind["params"]
                            if "params" in ind and isinstance(ind["params"], tuple)
                            else tuple()
                        )
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
            else:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, f"[i] Progress")
                    for ind in pbar:
                        getattr(self, ind)(*tuple(), **kwargs)
                else:
                    for ind in ta:
                        getattr(self, ind)(*tuple(), **kwargs)
                self._last_run = get_time(self.exchange, to_string=True)

        # Flush deferred appends for all/category modes.
        if not mode["custom"]:
            if self._pending_appends:
                new_df = pd.concat([self._df] + self._pending_appends, axis=1)
                # Swap the internal block manager so the original DataFrame
                # object is updated in-place (external references like user
                # variables keep working).
                self._df._mgr = new_df._mgr
                if hasattr(self._df, "_item_cache"):
                    self._df._item_cache.clear()
            self._pending_appends = None

        if verbose:
            logger.info("[i] Total indicators: %d", len(ta))
            logger.info(
                "[i] Columns added: %d", len(self._df.columns) - initial_column_count
            )
            logger.info("[i] Last Run: %s", self._last_run)
        if timed:
            logger.info("[i] Runtime: %s", final_time(stime))

        if returns:
            return self._df

    def ticker(self, ticker: str, **kwargs):
        """ticker

        This method downloads Historical Data if the package yfinance is installed.
        Additionally it can run a ta.Strategy; Builtin or Custom. It returns a
        DataFrame if there the DataFrame is not empty, otherwise it exits. For
        additional yfinance arguments, use help(ta.yf).

        Historical Data
        >>> df = df.ta.ticker("aapl")
        More specifically
        >>> df = df.ta.ticker("aapl", period="max", interval="1d", kind=None)

        Changing the period of Historical Data
        Period is used instead of start/end
        >>> df = df.ta.ticker("aapl", period="1y")

        Changing the period and interval of Historical Data
        Retrieves the past year in weeks
        >>> df = df.ta.ticker("aapl", period="1y", interval="1wk")
        Retrieves the past month in hours
        >>> df = df.ta.ticker("aapl", period="1mo", interval="1h")

        Show everything
        >>> df = df.ta.ticker("aapl", kind="all")

        Args:
            ticker (str): Any string for a ticker you would use with yfinance.
                Default: "SPY"
        Kwargs:
            kind (str): Options see above. Default: "history"
            ds (str): Data Source to use. Default: "yahoo"
            strategy (str | ta.Strategy): Which strategy to apply after
                downloading chart history. Default: None

            See help(ta.yf) for additional kwargs

        Returns:
            Exits if the DataFrame is empty or None
            Otherwise it returns a DataFrame
        """
        ds = kwargs.pop("ds", "yahoo")
        strategy = kwargs.pop("strategy", None)

        # Fetch the Data
        ds = ds.lower() if isinstance(ds, str) else ds
        # df = av(ticker, **kwargs) if ds and ds == "av" else yf(ticker, **kwargs)
        df = yf(ticker, **kwargs)

        if df is None:
            return
        elif df.empty:
            logger.error("[X] DataFrame is empty: %s", df.shape)
            return
        else:
            if kwargs.pop("lc_cols", False):
                df.index.name = df.index.name.lower()
                df.columns = df.columns.str.lower()
            self._df = df

        if strategy is not None:
            self.strategy(strategy, **kwargs)
        return df

    # Public DataFrame Methods: Indicators and Utilities
    #
    # Standard indicator wrappers are auto-generated via __getattr__ +
    # _make_indicator_method().  Only special-case methods that cannot
    # be auto-dispatched are defined explicitly below.

    def __getattr__(self, name: str):
        """Auto-dispatch to indicator functions without explicit wrapper methods.

        Any indicator registered in ``Category`` that does not have an explicit
        wrapper method defined on this class is wrapped on-the-fly via
        :func:`_make_indicator_method`.  The resulting bound method is cached on
        the class so that subsequent lookups bypass ``__getattr__`` entirely.
        """
        # Bail out immediately for private/dunder names to avoid recursion.
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # Confirm it is a known indicator; give a proper AttributeError otherwise.
        if not any(name in cats for cats in Category.values()):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        func = globals().get(name)
        if func is None or not callable(func):
            raise AttributeError(
                f"'{type(self).__name__}': indicator '{name}' is registered in "
                f"Category but not available in the module namespace"
            )
        method = _make_indicator_method(func)
        # Cache on the class so __getattr__ is not called again for this name.
        setattr(type(self), name, method)
        return method.__get__(self)

    # Momentum

    def inertia(
        self,
        length=None,
        rvi_length=None,
        scalar=None,
        refined=None,
        thirds=None,
        mamode=None,
        drift=None,
        offset=None,
        **kwargs,
    ):
        close = self._get_column(kwargs.pop("close", "close"))
        if refined is not None or thirds is not None:
            high = self._get_column(kwargs.pop("high", "high"))
            low = self._get_column(kwargs.pop("low", "low"))
            result = inertia(
                close=close,
                high=high,
                low=low,
                length=length,
                rvi_length=rvi_length,
                scalar=scalar,
                refined=refined,
                thirds=thirds,
                mamode=mamode,
                drift=drift,
                offset=offset,
                **kwargs,
            )
        else:
            result = inertia(
                close=close,
                length=length,
                rvi_length=rvi_length,
                scalar=scalar,
                refined=refined,
                thirds=thirds,
                mamode=mamode,
                drift=drift,
                offset=offset,
                **kwargs,
            )

        return self._post_process(result, **kwargs)

    def psl(
        self, open_=None, length=None, scalar=None, drift=None, offset=None, **kwargs
    ):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))

        close = self._get_column(kwargs.pop("close", "close"))
        result = psl(
            close=close,
            open_=open_,
            length=length,
            scalar=scalar,
            drift=drift,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)

    # Overlap

    def ichimoku(
        self,
        tenkan=None,
        kijun=None,
        senkou=None,
        include_chikou=True,
        offset=None,
        **kwargs,
    ):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result, span = ichimoku(
            high=high,
            low=low,
            close=close,
            tenkan=tenkan,
            kijun=kijun,
            senkou=senkou,
            include_chikou=include_chikou,
            offset=offset,
            **kwargs,
        )
        self._add_prefix_suffix(result, **kwargs)
        self._add_prefix_suffix(span, **kwargs)
        self._append(result, **kwargs)
        # return self._post_process(result, **kwargs), span
        return result, span

    def vwap(self, anchor=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))

        if not self.datetime_ordered:
            volume.index = self._df.index

        result = vwap(
            high=high,
            low=low,
            close=close,
            volume=volume,
            anchor=anchor,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)

    # Trend

    def long_run(self, fast=None, slow=None, length=None, offset=None, **kwargs):
        if fast is None and slow is None:
            return self._df
        else:
            result = long_run(
                fast=fast, slow=slow, length=length, offset=offset, **kwargs
            )
            return self._post_process(result, **kwargs)

    def short_run(self, fast=None, slow=None, length=None, offset=None, **kwargs):
        if fast is None and slow is None:
            return self._df
        else:
            result = short_run(
                fast=fast, slow=slow, length=length, offset=offset, **kwargs
            )
            return self._post_process(result, **kwargs)

    def psar(self, af0=None, af=None, max_af=None, offset=None, **kwargs):
        # close is genuinely optional in psar (used only as initial SAR seed).
        # Default to None (not "close") to preserve the original accessor behaviour.
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", None))
        result = psar(
            high=high,
            low=low,
            close=close,
            af0=af0,
            af=af,
            max_af=max_af,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)

    def tsignals(
        self,
        trend=None,
        asbool=None,
        trend_reset=None,
        trend_offset=None,
        offset=None,
        **kwargs,
    ):
        if trend is None:
            return self._df
        else:
            result = tsignals(
                trend,
                asbool=asbool,
                trend_offset=trend_offset,
                trend_reset=trend_reset,
                offset=offset,
                **kwargs,
            )
            return self._post_process(result, **kwargs)

    def xsignals(
        self,
        signal=None,
        xa=None,
        xb=None,
        above=None,
        long=None,
        asbool=None,
        trend_reset=None,
        trend_offset=None,
        offset=None,
        **kwargs,
    ):
        if signal is None:
            return self._df
        else:
            result = xsignals(
                signal=signal,
                xa=xa,
                xb=xb,
                above=above,
                long=long,
                asbool=asbool,
                trend_offset=trend_offset,
                trend_reset=trend_reset,
                offset=offset,
                **kwargs,
            )
            return self._post_process(result, **kwargs)

    # Utility

    def above(self, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        b = self._get_column(kwargs.pop("close", "b"))
        result = above(series_a=a, series_b=b, asint=asint, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def above_value(self, value=None, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        result = above_value(
            series_a=a, value=value, asint=asint, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def below(self, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        b = self._get_column(kwargs.pop("close", "b"))
        result = below(series_a=a, series_b=b, asint=asint, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    def below_value(self, value=None, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        result = below_value(
            series_a=a, value=value, asint=asint, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def cross(self, above=True, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        b = self._get_column(kwargs.pop("close", "b"))
        result = cross(
            series_a=a, series_b=b, above=above, asint=asint, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    def cross_value(self, value=None, above=True, asint=True, offset=None, **kwargs):
        a = self._get_column(kwargs.pop("close", "a"))
        # a = self._get_column(a, f"{a}")
        result = cross_value(
            series_a=a, value=value, above=above, asint=asint, offset=offset, **kwargs
        )
        return self._post_process(result, **kwargs)

    # Volatility

    # Volume

    def ad(self, open_=None, signed=True, offset=None, **kwargs):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = ad(
            high=high,
            low=low,
            close=close,
            volume=volume,
            open_=open_,
            signed=signed,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)

    def adosc(
        self, open_=None, fast=None, slow=None, signed=True, offset=None, **kwargs
    ):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = adosc(
            high=high,
            low=low,
            close=close,
            volume=volume,
            open_=open_,
            fast=fast,
            slow=slow,
            signed=signed,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)

    def cmf(self, open_=None, length=None, offset=None, **kwargs):
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        result = cmf(
            high=high,
            low=low,
            close=close,
            volume=volume,
            open_=open_,
            length=length,
            offset=offset,
            **kwargs,
        )
        return self._post_process(result, **kwargs)
