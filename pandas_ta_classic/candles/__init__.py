from .cdl_doji import cdl_doji
from .cdl_inside import cdl_inside
from .cdl_pattern import cdl_pattern, cdl, ALL_PATTERNS as CDL_PATTERN_NAMES
from .cdl_z import cdl_z
from .ha import ha

# Auto-import all native cdl_* pattern functions into module scope
# so they are accessible via ``from pandas_ta_classic.candles import cdl_engulfing``
# etc., but keep __all__ limited to the original set so that
# _build_category_dict() (which filters by __all__) does not expose
# them to the strategy runner until core.py has auto-dispatch (PR 6).
from .cdl_pattern import _NATIVE_PATTERNS as _np

for _name, _func in _np.items():
    globals()[f"cdl_{_name}"] = _func

# Clean up loop variables (may not exist if _np is empty)
del _np
for _v in ("_name", "_func"):
    globals().pop(_v, None)
del _v

__all__ = [
    "CDL_PATTERN_NAMES",
    "cdl",
    "cdl_doji",
    "cdl_inside",
    "cdl_pattern",
    "cdl_z",
    "ha",
]
