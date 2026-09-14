from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

# Names that live inside cdl_pattern.py rather than their own submodule.
install_lazy_subpackage(
    __name__,
    special={
        "cdl": ("cdl_pattern", "cdl"),
        "ALL_PATTERNS": ("cdl_pattern", "ALL_PATTERNS"),
    },
)
