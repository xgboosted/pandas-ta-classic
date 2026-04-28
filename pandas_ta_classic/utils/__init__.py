# -*- coding: utf-8 -*-
from ._candles import *
from ._core import *
from ._math import *
from ._signals import *
from ._time import *
from ._metrics import *
from .data import *

# Exclude the 'volatility' metrics function from the wildcard-export set so that
# 'from pandas_ta_classic.utils import *' (called in core.py) does not shadow
# the 'pandas_ta_classic.volatility' subpackage at the top-level namespace.
# The function remains accessible as pandas_ta_classic.utils.volatility().
__all__ = [_n for _n in dir() if not _n.startswith("_") and _n != "volatility"]
