import logging
from copy import copy
from dataclasses import dataclass, field
from multiprocessing import cpu_count, get_context
from time import perf_counter
from typing import Any, Optional
from warnings import simplefilter, warn

import pandas as pd
import numpy as np
from pandas.core.base import PandasObject

from pandas_ta_classic._meta import Category, EXCHANGE_TZ, Imports, version, _MATH_ALIASES
from pandas_ta_classic._indicator_loader import _find_indicator_func, _make_ta_wrapper
from pandas_ta_classic.utils import final_time, get_time, is_datetime_ordered, to_utc, total_time, yf

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
    ta: list = field(default_factory=list)  # Required.
    # Helpful. More descriptive version or notes or w/e.
    description: str = "TA Description"
    # Optional. Gets Exchange Time and Local Time execution time
    created: Optional[str] = field(default_factory=lambda: get_time(to_string=True))

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
        KC, et al will return a Pandas DataFrame. Ichimoku returns a single
        DataFrame for the known period; the forward-looking Span DataFrame is
        only available through the underlying ``pandas_ta.ichimoku()`` function.

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
    _last_run = get_time(_exchange, to_string=True)

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._df = pandas_obj
        self._last_run = get_time(self._exchange, to_string=True)

    @staticmethod
    def _validate(obj: tuple[pd.DataFrame, pd.Series]):
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
            self._last_run = get_time(self.exchange, to_string=True)  # Save when it completed it's run

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
        if value is not None and isinstance(value, str) and value in EXCHANGE_TZ:
            self._exchange = value

    @property
    def last_run(self) -> Optional[str]:
        """Returns the time when the DataFrame was last run."""
        return self._last_run

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

    def _indicators_by_category(self, name: str) -> Optional[list]:
        """Returns indicators by Categorical name."""
        return Category[name] if name in self.categories else None

    def _mp_worker(self, arguments: tuple):
        """Multiprocessing Worker to handle different Methods."""
        method, args, kwargs = arguments
        return getattr(self, method)(*args, **kwargs)

    def _post_process(self, result, **kwargs) -> tuple[pd.Series, pd.DataFrame]:
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        * In chain mode, auto-appends and returns the DataFrame for fluent chaining.
        """
        verbose = kwargs.pop("verbose", False)
        chain_mode = self._df.attrs.get("_ta_chain", False)

        if not isinstance(result, (pd.Series, pd.DataFrame)):
            if verbose:
                logger.error("The result was not a Series or DataFrame.")
            return self._df if chain_mode else self._df
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
                name, mode["all"] = name, True
            if arg.lower() in self.categories:
                name, mode["category"] = arg, True
        if isinstance(arg, Strategy):
            strategy_ = arg
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
        warn(
            "df.ta.constants() is deprecated and will be removed in a future "
            "release; adding horizontal charting lines is out of scope for a "
            "technical-analysis library. Assign the columns directly, e.g. "
            "df['0'] = 0.",
            FutureWarning,
            stacklevel=2,
        )
        if isinstance(values, (np.ndarray, list)):
            if append:
                for x in values:
                    self._df[f"{x}"] = x
                return self._df[self._df.columns[-len(values) :]]
            for x in values:
                del self._df[f"{x}"]

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
            "constants",
            "indicators",
            "strategy",
            "ticker",
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
            logger.info(f"{s}Abbreviations:\n    {', '.join(ta_indicators)}\n\nCandle Patterns:\n    {', '.join(ALL_PATTERNS)}")
        else:
            logger.info(s)

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
            logger.error("Not an available strategy.")
            return None

        # Remove Custom indicators with "length" keyword when larger than the DataFrame
        # Possible to have other indicator main window lengths to be included
        removal = []
        for kwds in ta:
            if isinstance(kwds, dict) and "length" in kwds and kwds["length"] > self._df.shape[0]:
                removal.append(kwds)
        if len(removal) > 0:
            for x in removal:
                ta.remove(x)

        verbose = kwargs.pop("verbose", False)
        if verbose:
            logger.info(f"Strategy: {name}\nIndicator arguments: {kwargs}")
            if mode["all"] or mode["category"]:
                excluded_str = ", ".join(excluded)
                logger.info(f"Excluded[{len(excluded)}]: {excluded_str}")

        timed = kwargs.pop("timed", False)
        results: Any = []
        use_multiprocessing = self.cores > 0
        has_col_names = False

        if timed:
            stime = perf_counter()

        if use_multiprocessing and mode["custom"]:
            # Determine if the Custom Model has 'col_names' parameter
            has_col_names = bool(len([True for x in ta if "col_names" in x and isinstance(x["col_names"], tuple)]))

            if has_col_names:
                use_multiprocessing = False

        if Imports["tqdm"]:
            # from tqdm import tqdm
            from tqdm import tqdm

        if use_multiprocessing:
            _total_ta = len(ta)

            # Create a lightweight copy of self that contains only the
            # original OHLCV columns.  Without this, each imap() call
            # pickles self._df (which grows as indicators are appended),
            # causing pandas BlockManager integrity errors in workers and
            # pool deadlocks.
            slim = copy(self)
            slim._df = self._df[self._df.columns[:initial_column_count]].copy()

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
                            (ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else ()),
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
            self._last_run = get_time(self.exchange, to_string=True)

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
                        params = ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else ()
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
                else:
                    for ind in ta:
                        params = ind["params"] if "params" in ind and isinstance(ind["params"], tuple) else ()
                        getattr(self, ind["kind"])(*params, **{**ind, **kwargs})
            else:
                if Imports["tqdm"] and verbose:
                    pbar = tqdm(ta, "[i] Progress")
                    for ind in pbar:
                        getattr(self, ind)(*(), **kwargs)
                else:
                    for ind in ta:
                        getattr(self, ind)(*(), **kwargs)
                self._last_run = get_time(self.exchange, to_string=True)

        if verbose:
            logger.info(f"Total indicators: {len(ta)}")
            logger.info(f"Columns added: {len(self._df.columns) - initial_column_count}")
            logger.info(f"Last Run: {self._last_run}")
        if timed:
            logger.info(f"Runtime: {final_time(stime)}")

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
            strategy (str | ta.Strategy): Which strategy to apply after
                downloading chart history. Default: None

            See help(ta.yf) for additional kwargs

        Returns:
            Exits if the DataFrame is empty or None
            Otherwise it returns a DataFrame
        """
        kwargs.pop("ds", None)
        strategy = kwargs.pop("strategy", None)

        # Fetch the Data
        df = yf(ticker, **kwargs)

        if df is None:
            return
        if df.empty:
            logger.error(f"DataFrame is empty: {df.shape}")
            return
        if kwargs.pop("lc_cols", False):
            df.index.name = df.index.name.lower()
            df.columns = df.columns.str.lower()
        self._df = df

        if strategy is not None:
            self.strategy(strategy, **kwargs)
        return df

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for private/dunder attributes
        if name.startswith("_"):
            raise AttributeError(name)
        func = _find_indicator_func(name)
        if func is None:
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
    # supports a legacy (visible, span) tuple return. This wrapper opts in to
    # the single-DataFrame return (as_dataframe=True) and forwards append_span,
    # so _post_process can handle it like any other indicator.
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
        close = self._get_column(kwargs.pop("close", "close"))
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
