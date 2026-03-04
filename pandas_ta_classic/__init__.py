name = "pandas-ta-classic"

"""
.. moduleauthor:: Kevin Johnson
"""
# Import metadata from _meta module to avoid circular imports
from pandas_ta_classic._meta import (
    Category,
    Imports,
    version,
    CANGLE_AGG,
    EXCHANGE_TZ,
    RATE,
)

# Core: accessor registration (side-effect) + public API
from pandas_ta_classic.core import (
    Strategy,
    AllStrategy,
    CommonStrategy,
    AnalysisIndicators,
)

# Indicators directly from subpackages (no longer routed through core)
from pandas_ta_classic.candles import *
from pandas_ta_classic.cycles import *
from pandas_ta_classic.momentum import *
from pandas_ta_classic.overlap import *
from pandas_ta_classic.performance import *
from pandas_ta_classic.statistics import *
from pandas_ta_classic.trend import *
from pandas_ta_classic.volatility import *
from pandas_ta_classic.volume import *
from pandas_ta_classic.utils import *

__version__ = version
__description__ = (
    "An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. "
    "Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib. "
    "This is the classic/community maintained version."
)

__all__ = [
    "Category",
    "Imports",
    "version",
    "CANGLE_AGG",
    "EXCHANGE_TZ",
    "RATE",
    "Strategy",
    "AllStrategy",
    "CommonStrategy",
    "AnalysisIndicators",
]
