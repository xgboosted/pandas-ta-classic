from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

# Names that live inside cdl_pattern.py rather than their own submodule.
install_lazy_subpackage(
    __name__,
    special={
        "cdl": ("cdl_pattern", "cdl"),
        "ALL_PATTERNS": ("cdl_pattern", "ALL_PATTERNS"),
    },
)

if TYPE_CHECKING:
    from .cdl_doji import cdl_doji as cdl_doji
    from .cdl_inside import cdl_inside as cdl_inside
    from .cdl_pattern import ALL_PATTERNS as ALL_PATTERNS, cdl as cdl, cdl_pattern as cdl_pattern
    from .cdl_z import cdl_z as cdl_z
    from .ha import ha as ha
