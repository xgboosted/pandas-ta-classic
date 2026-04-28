name = "pandas-ta-classic"

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

# Re-expose the volatility subpackage under its own name.
# The utils.metrics.volatility() function (same name) would otherwise shadow it
# after the wildcard import above.
from . import volatility  # noqa: F811

__version__ = version
__description__ = (
    "An easy to use Python 3 Pandas Extension providing a comprehensive set of Technical Analysis indicators."
    "Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."
    "This is the classic/community maintained version."
)
