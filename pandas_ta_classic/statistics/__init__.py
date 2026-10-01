from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .beta import beta as beta
    from .correl import correl as correl
    from .entropy import entropy as entropy
    from .kurtosis import kurtosis as kurtosis
    from .mad import mad as mad
    from .md import md as md
    from .median import median as median
    from .quantile import quantile as quantile
    from .skew import skew as skew
    from .stderr import stderr as stderr
    from .stdev import stdev as stdev
    from .tos_stdevall import tos_stdevall as tos_stdevall
    from .variance import variance as variance
    from .zscore import zscore as zscore
