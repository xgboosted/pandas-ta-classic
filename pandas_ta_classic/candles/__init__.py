from .cdl_doji import cdl_doji
from .cdl_inside import cdl_inside
from .cdl_pattern import cdl_pattern, cdl, ALL_PATTERNS as CDL_PATTERN_NAMES
from .cdl_z import cdl_z
from .ha import ha

# Auto-import all native cdl_* pattern functions
from .cdl_pattern import _NATIVE_PATTERNS as _np
import importlib as _importlib

_all_cdl = [
    "CDL_PATTERN_NAMES",
    "cdl",
    "cdl_doji",
    "cdl_inside",
    "cdl_pattern",
    "cdl_z",
    "ha",
]

for _name, _func in _np.items():
    _fname = f"cdl_{_name}"
    globals()[_fname] = _func
    _all_cdl.append(_fname)

__all__ = sorted(set(_all_cdl))

del _np, _importlib, _name, _func, _fname, _all_cdl
