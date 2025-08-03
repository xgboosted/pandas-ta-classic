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

# Import core functionality
from pandas_ta_classic.core import *

__version__ = version
__description__ = (
    "An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. "
    "Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib. "
    "This is the classic/community maintained version."
)
